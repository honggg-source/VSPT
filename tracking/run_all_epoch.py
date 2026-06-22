import os
import sys
import argparse
import shutil
import yaml
import subprocess
import time

prj_path = os.path.join(os.path.dirname(__file__), '..')


def main():
    parser = argparse.ArgumentParser(description='Batch runner using subprocess for speed.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method (e.g. ostrack).')
    parser.add_argument('tracker_param', type=str, help='Name of config file (e.g. vitb_256_mae_32x4_ep60).')
    parser.add_argument('--dataset_name', type=str, default='lsotb', help='Name of dataset.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)

    # 核心参数：测试范围
    parser.add_argument('--start_epoch', type=int, default=7, help='Start epoch.')
    parser.add_argument('--end_epoch', type=int, default=8, help='End epoch.')

    args = parser.parse_args(['ostrack','vitb_256_mae_32x4_ep60'])

    # 1. 定位 Yaml 文件
    yaml_rel_path = os.path.join('experiments', args.tracker_name, args.tracker_param + '.yaml')
    yaml_path = os.path.join(prj_path, yaml_rel_path)
    backup_path = yaml_path + '.bak'

    if not os.path.exists(yaml_path):
        print(f"Error: Yaml file not found at {yaml_path}")
        return

    # 2. 备份 Yaml
    if os.path.exists(backup_path):
        print("Warning: A backup file (.bak) was found. Restoring it first.")
        shutil.copy(backup_path, yaml_path)
    print(f"Backing up config to {backup_path}")
    shutil.copy(yaml_path, backup_path)

    try:
        # 读取原始配置结构
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # 3. 循环调用
        for ep in range(args.start_epoch, args.end_epoch + 1):
            print(f"\n" + "=" * 60)
            print(f"   STARTING PROCESS FOR EPOCH {ep}")
            print(f"=" * 60)

            # --- 修改 YAML ---
            if 'TEST' not in config_data:
                config_data['TEST'] = {}
            config_data['TEST']['EPOCH'] = ep

            with open(yaml_path, 'w') as f:
                yaml.dump(config_data, f)

            # --- 构造命令行命令 ---
            # 这里的 runid 设为 ep，保证结果保存在 output/test/.../ep/ 下
            cmd = [
                sys.executable, 'test.py',
                args.tracker_name,
                args.tracker_param,
                '--dataset_name', args.dataset_name,
                '--runid', str(ep),
                '--debug', str(args.debug),
                '--threads', str(args.threads),
                '--num_gpus', str(args.num_gpus)
            ]

            if args.sequence is not None:
                cmd.extend(['--sequence', str(args.sequence)])

            # --- 执行命令 (独立进程，满速运行) ---
            print(f"Executing: {' '.join(cmd)}")
            start_time = time.time()

            # check=True 会在 test.py 报错时抛出异常停止脚本，防止无效空跑
            subprocess.run(cmd, check=True)

            elapsed = time.time() - start_time
            print(f"Epoch {ep} finished in {elapsed:.1f} seconds.")

    except subprocess.CalledProcessError:
        print("\n❌ Error: The testing subprocess failed. Stopping batch.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        # 4. 还原 YAML
        if os.path.exists(backup_path):
            print(f"\nRestoring original yaml configuration...")
            shutil.copy(backup_path, yaml_path)
            os.remove(backup_path)
            print("Done.")


if __name__ == '__main__':
    main()