from argparse import ArgumentParser
import math

def parse_guidance_args():
    parser = ArgumentParser()
    
    parser.add_argument("--num_div", type=int, default=64)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha_r", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--Phi_init", type=float, default=math.radians(45.0))
    parser.add_argument("--Phi_min", type=float, default=math.radians(15.0))
    parser.add_argument("--Phi_max", type=float, default=math.radians(75.0))
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--is_peptide", type=bool, default=True)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--target_protein", type=str, default="AAAAA")
    parser.add_argument("--target_enhancer_class", type=int, default=0)
    parser.add_argument("--target_DNA_shape", type=str, default='HelT')
    
    args = parser.parse_args()
    return args