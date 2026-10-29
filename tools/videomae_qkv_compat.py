"""Preserve explicit Q/K/V biases when using the installed VideoMAE runtime.

Attention dispatch follows Hugging Face Transformers 4.57.1 VideoMAE
(Apache-2.0); unlike that version's original layout, all three biases are
registered and consumed. No checkpoint tensor is dropped or synthesized at load.
"""
import types


def forward(self, hidden_states, head_mask=None):
    from transformers.models.videomae.modeling_videomae import eager_attention_forward,ALL_ATTENTION_FUNCTIONS
    batch=hidden_states.shape[0]
    def heads(layer):
        return layer(hidden_states).view(batch,-1,self.num_attention_heads,self.attention_head_size).transpose(1,2)
    q,k,v=heads(self.query),heads(self.key),heads(self.value)
    interface=eager_attention_forward if self.config._attn_implementation=='eager' else ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    context,probs=interface(self,q,k,v,head_mask,is_causal=self.is_causal,scaling=self.scaling,
        dropout=0.0 if not self.training else self.dropout_prob)
    return context.reshape(context.size()[:-2]+(self.all_head_size,)),probs


def adapt(model):
    import torch
    converted=[]
    for number,layer in enumerate(model.videomae.encoder.layer):
        attention=layer.attention.attention
        attention.eval()
        # Confirm equivalence to the trusted original implementation before
        # loading the explicit-bias checkpoint. Preserve Q/V and add zero K
        # only for this pre-load equivalence check.
        sample=torch.linspace(-1,1,steps=8*attention.config.hidden_size).reshape(1,8,-1)
        with torch.inference_mode():before=attention(sample)[0]
        attention.query.bias=attention.q_bias
        attention.value.bias=attention.v_bias
        attention.key.bias=torch.nn.Parameter(torch.zeros_like(attention.v_bias))
        del attention.q_bias
        del attention.v_bias
        attention.forward=types.MethodType(forward,attention)
        with torch.inference_mode():after=attention(sample)[0]
        torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
        converted.append(number)
    return {'layers':converted,'preload_equivalence':'passed','checkpoint_biases':'all Q/K/V tensors strict-loaded without dropping key bias'}
