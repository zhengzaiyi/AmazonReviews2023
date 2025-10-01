cd seq_rec_results/dataset/
# TODO: constant larger datasets
python process_amazon_2023.py \
    --domain  Musical_Instruments \
    --device cuda:0 \
    --plm bert-base-uncased
    