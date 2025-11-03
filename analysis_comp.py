import pickle as pkl
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completions_path', type=str, default='completions/completions_ml-1m_use_hf_local_do_test_do_test_rl_use_vllm.pkl')
    return parser.parse_args()

def main():
    args = parse_args()
    completions = pkl.load(open(args.completions_path, "rb"))
    print(completions)
    stats = {}
    for i, completion in enumerate(completions):
        for recaller, params in completion.items():
            if recaller not in stats:
                stats[recaller] = {
                    'top-k': [0.0] * i,
                    'score-weight': [0.0] * i,
                }
            
            stats[recaller]['top-k'].append(params['top-k'])
            stats[recaller]['score-weight'].append(params['score-weight'])

if __name__ == "__main__":
    main()