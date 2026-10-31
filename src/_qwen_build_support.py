"""Keep the stable wheel limited to its historical S081 runtime."""

from pathlib import Path

from setuptools.command.build_py import build_py


class StableBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            item
            for item in super().find_package_modules(package, package_dir)
            if not item[1].startswith("hypergan_hndl")
        ]

    def find_data_files(self, package, src_dir):
        return [
            path
            for path in super().find_data_files(package, src_dir)
            if Path(path).name != "hypergan-hndl.lock.json"
        ]

    def run(self):
        super().run()
        # An older candidate build may have left files in build/lib. Discovery
        # exclusions alone do not remove them before install_lib copies the tree.
        root = Path(self.build_lib) / "qwen_image_sliders"
        for pattern in ("hypergan_hndl*.py*", "__pycache__/hypergan_hndl*.pyc", "hypergan-hndl.lock.json"):
            for path in root.glob(pattern):
                path.unlink()
