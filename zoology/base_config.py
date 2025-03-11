import argparse
from datetime import datetime
from functools import partial
from typing import List, Tuple, Union

from pydantic import BaseModel


from zoology.utils import import_from_str


class BaseConfig(BaseModel):
    @classmethod
    def from_cli(cls):
        import yaml
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--config', type=str, default=None, help='Path to the config file')
        parser.add_argument('--run_id', type=str, default=None, help='Run ID for the training')
        args, extra_args = parser.parse_known_args()


        if args.config is not None:
            with open(args.config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}
        
        # Override with any extra arguments from the command line
        def _nested_update(config, args):
            for key, value in args.items():
                keys = key.split(".")
                for key in keys[:-1]:
                    config = config.setdefault(key, {})
                config[keys[-1]] = value

        extra_args = dict([arg.lstrip("-").split("=") for arg in extra_args])
        extra_args = {k.replace("-", "_"): v for k, v in extra_args.items()}
        _nested_update(config, extra_args)
        config = cls.parse_obj(config)

        if config.run_id is None:
            config.run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return config

    def print(self):
        try:
            import rich
            rich.print(self)
        except ImportError:
            print(self)


class FunctionConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)

class ModuleConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)

