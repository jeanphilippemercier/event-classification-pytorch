import kfp
from kfp import components as comp


def load_uquake(test: int) -> int:
    from uquake.core import stream

    return test * 2


load_uquake_op = comp.func_to_container_op(load_uquake, base_image=
                                           'tensorflow/tensorflow:1.14.0-py3')

