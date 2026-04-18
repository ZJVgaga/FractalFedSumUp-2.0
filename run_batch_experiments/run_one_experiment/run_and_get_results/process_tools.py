import io
import pickle


def measure_size(obj):
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    return buffer.getbuffer().nbytes  

def early_stopper(patience=3):
    wait = 0  # 当前等待的轮次数
    expectation=0.01
    def should_early_stop(rounds, evaluate_acc, round_accuracies):
        nonlocal wait
        if rounds >= patience:
            threshold = round_accuracies[rounds - patience] +expectation
            if evaluate_acc < threshold:
                wait += 1
                # 打印准确率未达到阈值的信息
                print(f"Round {rounds} accuracy ({evaluate_acc}) did not reach the threshold ({threshold}), waiting for {wait} rounds.")
            else:
                wait = 0
                # 新增：打印准确率达到或超过阈值的信息
                print(f"Round {rounds} accuracy ({evaluate_acc}) has reached or exceeded the threshold ({threshold}), resetting wait counter.")
            
            if wait == patience:
                print(f"Early stopping triggered at round {rounds}")
                return True
        return False
    return should_early_stop