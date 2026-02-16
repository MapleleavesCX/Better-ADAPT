import numpy as np


def find_inhomogeneous_parts(covered):
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected:")
        for i, shape in enumerate(shapes):
            if shape != shapes[0]:  # 假设第一个子数组的形状是标准形状
                print(f"Element {i}: shape={shape}, content={covered[i]}")
    else:
        print("All elements have the same shape.")



def print_inhomogeneous_details(covered):
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected. Details:")
        for i, arr in enumerate(covered):
            print(f"Element {i}: shape={arr.shape}, dtype={arr.dtype}, content={arr}")
    else:
        print("All elements have the same shape.")


def check_and_fix_inhomogeneous(covered):
    # 检测不规则的部分
    shapes = [arr.shape for arr in covered]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print("Inhomogeneous parts detected. Fixing...")
        
        # 填充数组
        max_length = max(len(arr) for arr in covered)
        padded_covered = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=False) for arr in covered]
        
        return padded_covered
    else:
        print("All elements have the same shape.")
        return covered

if __name__ == '__main__':
    pass
