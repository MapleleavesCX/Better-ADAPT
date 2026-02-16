import numpy as np
from adapt.metric.metric import Metric


class SignSignCoverage(Metric):
    def __init__(self):
        """
        初始化 Sign-Sign Coverage 计算器。
        :param history: 保存历史激活模式的最大长度。
        """

        self._internals = []
        super(SignSignCoverage, self).__init__()

    def covered(self, internals, **kwargs):
        """
        计算当前输入是否满足 SSC 覆盖标准。
        :param internals: 当前输入的各层激活值（列表形式）。
        :return: 每一层满足 SSC 标准的布尔数组。
        """
        covered = []

        if not self._internals:
            # 如果是第一次调用，初始化历史记录
            for layer_output in internals:
                self._internals.append(np.array([layer_output]))
                # 初始情况下，所有神经元都未覆盖
                vec = [False] * len(layer_output)
                covered.append(np.array(vec))
        else:
            # 遍历每一层，计算 SSC 覆盖
            for index, layer_output in enumerate(internals):
                # 获取历史记录中的符号变化
                historical_signs = np.sign(self._internals[index])[0]
                current_signs = np.sign(layer_output)
                # 检查符号变化是否满足 SSC 条件
                vec = historical_signs != current_signs
                covered.append(np.array(vec))
                # 更新历史记录
                self._internals[index] = layer_output

        return np.array(covered, dtype=object)

    def __repr__(self):
        return 'SignSignCoverage()'


# 示例使用
if __name__ == "__main__":
    # 创建度量对象
    metric = SignSignCoverage()

    # 初始化历史覆盖状态
    old_covered = [False] * 11
    old_covered = np.array(old_covered)

    # 模拟多次输入
    for i in range(5):
        # 随机生成三层网络的激活值
        layout1 = np.random.randn(3)  # 第一层有 3 个神经元
        layout2 = np.random.randn(6)  # 第二层有 6 个神经元
        layout4 = np.random.randn(2)  # 第三层有 2 个神经元
        internals_example = [layout1, layout2, layout4]

        print(f"Internals Example: {internals_example}")

        # 计算 SSC 覆盖布尔数组
        new_covered = np.concatenate(metric.covered(internals_example))
        old_covered = np.bitwise_or(old_covered, new_covered)

        print(f"Coverages: {np.mean(old_covered)}\n\n")