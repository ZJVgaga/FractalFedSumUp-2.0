import io
import pickle


def measure_size(obj):
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    return buffer.getbuffer().nbytes  

def early_stopper(patience=3):
    wait = 0  # Current number of waiting rounds
    expectation=0.01
    def should_early_stop(rounds, evaluate_acc, round_accuracies):
        nonlocal wait
        if rounds >= patience:
            threshold = round_accuracies[rounds - patience] +expectation
            if evaluate_acc < threshold:
                wait += 1
                # Print information when accuracy does not reach the threshold
                print(f"Round {rounds} accuracy ({evaluate_acc}) did not reach the threshold ({threshold}), waiting for {wait} rounds.")
            else:
                wait = 0
                # New: Print information when accuracy reaches or exceeds the threshold
                print(f"Round {rounds} accuracy ({evaluate_acc}) has reached or exceeded the threshold ({threshold}), resetting wait counter.")
            
            if wait == patience:
                print(f"Early stopping triggered at round {rounds}")
                return True
        return False
    return should_early_stop