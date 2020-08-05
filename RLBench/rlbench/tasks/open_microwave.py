from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition


class OpenMicrowave(Task):

    def init_task(self) -> None:
        self.register_success_conditions([JointCondition(
            Joint('microwave_door_joint'), np.deg2rad(80))])

    def init_episode(self, index: int) -> List[str]:
        return ['open microwave',
                'open the microwave door',
                'pull the microwave door open']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 4.], [0, 0, 3.14 / 4.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
