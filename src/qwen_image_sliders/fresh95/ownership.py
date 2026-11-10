"""Original full core/PG guards, installed package ownership and retryable cleanup."""
from copy import deepcopy
import base64,hashlib,importlib.util
from importlib.metadata import distribution
from pathlib import Path
from .locked_sources import ORIGINAL
from .util import source_record


def require():
    from qwen_image_sliders.core_profile import require_current_core
    require_current_core()  # Unchanged complete core577/PG68 Source verification.
    dist=distribution('qwen-image-particle-sliders')
    root=Path(dist.locate_file('qwen_image_sliders')).absolute()
    if dist.version!='0.3.0' or Path(__file__).absolute().parent!=root/'fresh95':
        raise RuntimeError('new namespace must belong to the original installed product distribution')
    for name,pin in ORIGINAL.items():
        raw=(root/name).read_bytes()
        if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
            raise RuntimeError('captured original Qwen Source differs: '+name)
    expected={str(p):p for p in(dist.files or())if str(p).startswith('qwen_image_sliders/fresh95/')and p.suffix=='.py'}
    actual={str(p.relative_to(root.parent)):p for p in(root/'fresh95').rglob('*.py')if '__pycache__'not in p.parts}
    if set(expected)!=set(actual):raise RuntimeError('new namespace/installed RECORD inventory differs')
    for name,path in actual.items():
        pin=expected[name].hash
        digest=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
        if pin is None or pin.mode!='sha256' or pin.value!=digest:
            raise RuntimeError('new namespace differs from genuine installed RECORD')
    spec=importlib.util.find_spec('qwen_image_sliders.fresh95')
    if spec is None or Path(spec.origin).absolute()!=root/'fresh95/__init__.py':
        raise RuntimeError('new namespace is shadowed')


def sources():
    import inspect
    from diffusers import QwenImage21Pipeline,QwenImage21Transformer2DModel
    from qwen_image_sliders import adapters,train,runtime,lifecycle,tiny,contracts,core_profile,learning_rates
    from particle_sliders import pg09,pg09_images,pg09_backends,pg09_readout_critic,qwen_native
    from particlegan import init
    modules=(('adapters',adapters),('train',train),('runtime',runtime),('lifecycle',lifecycle),
        ('tiny',tiny),('contracts',contracts),('core_profile',core_profile),('learning_rates',learning_rates),
        ('pg09',pg09),('pg09_images',pg09_images),('pg09_backends',pg09_backends),
        ('readout_critic',pg09_readout_critic),('qwen_native',qwen_native),('PG095_init',init))
    original={name:source_record(module.__file__)for name,module in modules}
    original.update(native_pipeline=source_record(inspect.getsourcefile(QwenImage21Pipeline)),
        native_transformer=source_record(inspect.getsourcefile(QwenImage21Transformer2DModel)))
    own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))}
    return dict(own=own,original=original)


_MISSING=object()
class Owner:
    """Borrow runtime; remove only owned hooks, then restore settings, then release.

    No model unload is implied for a borrowed runtime. A transferred release
    callback, if provided, runs only after hook/settings cleanup succeeds.
    Failed operations retain handles and this owner for explicit cleanup retry.
    """
    def __init__(self,runtime,release=None):
        if release is not None and not callable(release):raise TypeError('optional owned release must be callable')
        self.runtime,self.bank,self.release=runtime,None,release
        self.closed=self.closing=self.cleanup_pending=False
        self._release_pending=release is not None;self._operations=[];self._sealed=False
        self._handles_captured=False;self._handle_container=None
        self._owned_handles=();self._pending_handles=[];self._hook_callbacks={}
        root=runtime.model
        self._before=[(m,k,m.__dict__.get(k,_MISSING))for m in root.modules()
            for k in('training','gradient_checkpointing','_gradient_checkpointing_func')]
        self._before +=[(p,'requires_grad',p.requires_grad)for p in root.parameters()]
        self._before.append((root.pos_embed,'freqs',root.pos_embed.freqs))
    def capture_handles(self):
        """Called in the original constructor's finally, before attachment escapes.

        A partial original constructor may have already removed some hooks. Its
        remaining actual handle identities are captured once; a later public
        bank.handles mutation can never expand our cleanup authority.
        """
        if self._handles_captured:return
        from torch.utils.hooks import RemovableHandle
        container=getattr(self.bank,'handles',None)if self.bank is not None else None
        if container is None:
            self._handles_captured=True;return
        if type(container)is not list or any(type(h)is not RemovableHandle for h in container):
            raise RuntimeError('original constructor handle container/type differs')
        owned=tuple(container)
        if len({id(h)for h in owned})!=len(owned):raise RuntimeError('duplicate constructor handle identity')
        callbacks={};pending=[]
        for handle in owned:
            hooks=handle.hooks_dict_ref()
            if hooks is not None and handle.id in hooks:
                callbacks[id(handle)]=hooks[handle.id];pending.append(handle)
        self._handle_container=container;self._owned_handles=owned
        self._pending_handles=pending;self._hook_callbacks=callbacks
        # Prune handles the original partial-constructor cleanup already removed.
        container[:]=pending;self._handles_captured=True
    def _check_handles(self):
        if self._handle_container is None:return
        current=getattr(self.bank,'handles',None)
        if(current is not self._handle_container or len(current)!=len(self._pending_handles)
                or any(a is not b for a,b in zip(current,self._pending_handles))):
            raise RuntimeError('foreign appended/replaced handle container retained; restore original owned pending list to retry')
        for handle in self._pending_handles:
            hooks=handle.hooks_dict_ref()
            if(hooks is not None and handle.id in hooks
                    and hooks[handle.id]is not self._hook_callbacks[id(handle)]):
                raise RuntimeError('foreign replacement hook callback retained')
    def seal(self):
        if self._sealed:return
        for owner,key,original in self._before:
            installed=owner.requires_grad if key=='requires_grad'else owner.__dict__.get(key,_MISSING)
            if installed is not original or key=='freqs':self._operations.append([owner,key,original,installed])
        self._sealed=True
    def observe_native(self):
        # Native forwards legitimately rebind the unregistered RoPE device cache.
        # Called only immediately after an owned operation, including its failure.
        self.seal()
        for operation in self._operations:
            if operation[1]=='freqs':operation[3]=operation[0].__dict__.get('freqs',_MISSING)
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('Qwen owner closing/closed; only cleanup retry allowed')
        self._check_handles()
        for owner,key,original,installed in self._operations:
            current=owner.requires_grad if key=='requires_grad'else owner.__dict__.get(key,_MISSING)
            if current is not installed:raise RuntimeError('foreign native setting retained before operation: '+key)
    def close(self):
        if self.closed:return
        self.closing=True;self.seal();errors=[]
        # Cleanup authority is the original captured identities, never the
        # mutable public bank.handles list. Refuse foreign changes before removal.
        try:
            self.capture_handles();self._check_handles()
            pending=[]
            for handle in self._pending_handles:
                hooks=handle.hooks_dict_ref()
                if hooks is None or handle.id not in hooks:continue
                try:handle.remove()
                except BaseException as error:pending.append(handle);errors.append(error)
            self._pending_handles=pending
            if self._handle_container is not None:self._handle_container[:]=pending
        except BaseException as error:errors.append(error)
        if not errors:
            pending=[]
            for owner,key,original,installed in self._operations:
                try:
                    current=owner.requires_grad if key=='requires_grad'else owner.__dict__.get(key,_MISSING)
                    if current is not original:
                        if current is not installed:raise RuntimeError('foreign setting retained: '+key)
                        if key=='requires_grad':owner.requires_grad_(original)
                        elif original is _MISSING:delattr(owner,key)
                        else:setattr(owner,key,original)
                except BaseException as error:pending.append([owner,key,original,installed]);errors.append(error)
            self._operations=pending
        if not errors and self._release_pending:
            try:self.release();self._release_pending=False
            except BaseException as error:errors.append(error)
        self.cleanup_pending=bool(errors)
        if errors:
            failure=BaseExceptionGroup('Qwen cleanup pending; retry cleanup_owner.close()',errors)
            failure.cleanup_owner=self;raise failure
        self.closed,self.closing=True,False
