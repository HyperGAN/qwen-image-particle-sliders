"""Native bank/critic initialization only; shared PG owns every update/loss."""
def build_game(runtime, records, stamp, declared, seed, *, steps, preset='bispf_optimized',
               lr=None, d_lr_mult=None, prior_lr_mult=None, critic_kind='readout'):
    import torch
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter, image_condition_features
    from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
    from qwen_image_sliders.adapters import AdapterBank
    from qwen_image_sliders.learning_rates import resolve_alias
    from qwen_image_sliders.train import response_rows
    from .execution import initialization
    from .native import require_live
    require_live(runtime)
    rates=resolve_alias(lr=lr,d_lr_mult=d_lr_mult,prior_lr_mult=prior_lr_mult)
    edits=[]
    for row in records:
        delta=row['target']-row['neutral']
        if not torch.isfinite(delta).all():raise ValueError('nonfinite native teacher delta')
        rms=float(delta.double().square().mean().sqrt())
        if not row['control'] and rms>1e-8:edits.append(rms)
    if len(edits)<2:raise ValueError('two actual nondegenerate edit rows required')
    response_scale=float(torch.tensor(edits,dtype=torch.float64).median())
    bank=None
    try:
        with initialization(seed,device=str(runtime.device)):
            bank=AdapterBank(runtime.model,stamp).to(runtime.device)
            particles=torch.nn.Parameter(torch.randn(int(stamp.spec['parts']),int(stamp.spec['particle_dim']),device=runtime.device)*.02)
            adapter=QwenImage21ResponseAdapter(runtime,bank,particles,model_identity=runtime.base_identity)
            prepared=[adapter.prepare(row) for row in response_rows(runtime,records)]
            conditions=image_condition_features(prepared)
            critic_class=ConditionalReadoutCritic
            if critic_kind=='hndl':
                from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic
                critic_class=HNDLConditionalReadoutCritic
            elif critic_kind!='readout':raise ValueError('unsupported critic')
            critic=critic_class(prepared[0].target.numel(),conditions.shape[1]).to(runtime.device)
            game=ConditionalPG09(adapter,prepared,critic,conditions=conditions,particles=particles,
                response_domain='continuous_field',response_scale=response_scale,preset=preset,
                total_steps=steps,batch_size=int(declared['adv_batch']),seed=seed,
                recipe_overrides=rates,sampling='balanced')
        game.normalization_records=len(edits)
        return game
    except BaseException:
        if bank is not None:bank.close()
        raise
