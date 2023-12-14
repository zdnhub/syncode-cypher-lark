import torch
import os
from transformers import LlamaForCausalLM
import matplotlib.pyplot as plt

model = LlamaForCausalLM.from_pretrained("/share/models/hugging_face/CodeLlama-7b", torch_dtype=torch.bfloat16).eval()

attn_layers = [model.retrieve_modules_from_names(['model.layers.' + str(i) + '.self_attn.'])[0] for i in range(32)]

for layer_num, attn_layer in enumerate(attn_layers):
    q_proj = attn_layer.q_proj.weight.to(torch.float)
    k_proj = attn_layer.k_proj.weight.to(torch.float)

    num_heads = attn_layer.num_heads
    d_k = q_proj.shape[0] // num_heads

    for head_num in range(num_heads):
        q = q_proj[head_num * d_k : (head_num + 1) * d_k]
        k = k_proj[head_num * d_k : (head_num + 1) * d_k]

        # Compute the eigenvalues of the projection matrix
        print(q.shape, k.shape)
        eg = torch.linalg.eig(q @ k.T)
        print(eg.eigenvalues.shape)

        # Normalize the eigenvalues
        max_norm = torch.max(torch.abs(eg.eigenvalues))
        normalized_eg = eg.eigenvalues/max_norm

        # Visualize the complex tensor        
        fig, ax = plt.subplots()
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        real = normalized_eg.real.detach().numpy()
        imag = normalized_eg.imag.detach().numpy()
        ax.scatter(real, imag)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title('Eigenvalues of Q projection matrix')
        os.makedirs('plots', exist_ok=True)
        fig.savefig('plots/eig_' + str(layer_num) + '_' + str(head_num) + '.png')
        plt.close(fig)
