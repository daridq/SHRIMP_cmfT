# SHRIMP_Flowmatch
A version of U-Net Diffusion with added architecture for Flow Matching


### Flow Matching + Transformer
- How to run model on gadi: `python cfm_dit_cmd_generate full`

Note that:
- for transformer model, if `future_frame` and `history frame` are both set to 0, then the performance would be extremely poor (no temporal info is used).
- `batch_size` must be even number
- change the `dit_model` size accordingly
- for fastest run: `python cfm_dit_cmd_generate quick`
