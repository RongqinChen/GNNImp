

for((seed=0;seed<5;seed=seed+1));
do 
    python main.py --dataset P12 --seed $((seed)) --model_name TransformerModelV2 
    python main.py --dataset P12 --seed $((seed)) --model_name TransformerModel 
    python main.py --dataset P12 --seed $((seed)) --model_name TCN 
    python main.py --dataset P12 --seed $((seed)) --model_name GRU 
    python main.py --dataset P12 --seed $((seed)) --model_name SAND 
    python main.py --dataset P12 --seed $((seed)) --model_name Raindrop 
    python main.py --dataset P12 --seed $((seed)) --model_name MTGNN 
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP 
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg8 
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg16 
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg24 

    python main.py --dataset P12 --seed $((seed)) --model_name TransformerModelV2  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name TransformerModel  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name TCN  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name GRU  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name SAND  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name Raindrop  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name MTGNN  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg8  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg16  --with_missing_ratio
    python main.py --dataset P12 --seed $((seed)) --model_name SVMTP_Seg24  --with_missing_ratio

    python main.py --dataset P19 --seed $((seed)) --model_name TransformerModelV2 
    python main.py --dataset P19 --seed $((seed)) --model_name TransformerModel 
    python main.py --dataset P19 --seed $((seed)) --model_name TCN 
    python main.py --dataset P19 --seed $((seed)) --model_name GRU 
    python main.py --dataset P19 --seed $((seed)) --model_name SAND 
    python main.py --dataset P19 --seed $((seed)) --model_name Raindrop 
    python main.py --dataset P19 --seed $((seed)) --model_name MTGNN 
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP 
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg8 
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg16 
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg24 

    python main.py --dataset P19 --seed $((seed)) --model_name TransformerModelV2  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name TransformerModel  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name TCN  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name GRU  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name SAND  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name Raindrop  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name MTGNN  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg8  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg16  --with_missing_ratio
    python main.py --dataset P19 --seed $((seed)) --model_name SVMTP_Seg24  --with_missing_ratio
done
