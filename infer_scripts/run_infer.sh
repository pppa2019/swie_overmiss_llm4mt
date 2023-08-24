work_dir=
cd $work_dir
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
plm_dir=`the dirtory of you fine-tuned models`
base_model=`the model path`
mkdir -p inference_output/${base_model}
mkdir -p log/${base_model}
inference()
{
    src=$1
    tgt=$2
    step=$3
    test_type=$4
    
    python3 train/inference.py --model-name-or-path $plm_dir/$base_model/checkpoint-$step \
        -lp $src-$tgt \
        -t 0.1 \
        -b 4 \
        -tp 1 \
        -sa 'beam' \
        -ins test/instruct_inf.txt \
        -i test/WMT22/newstest22.${test_type}$src-$tgt.$src \
        -o inference_output/${base_model}/$step.$src-$tgt.$tgt.${test_type}none-hint.txt \
        2>&1 | tee log/${base_model}/$src-$tgt.${test_type}none-hint.log
}


for step in  600
do 
    (export CUDA_VISIBLE_DEVICES=0;inference de en $step)& \
    (export CUDA_VISIBLE_DEVICES=1;inference en de $step)& \
    (export CUDA_VISIBLE_DEVICES=2;inference en zh $step)& \
    (export CUDA_VISIBLE_DEVICES=3;inference zh en $step)& \
    (export CUDA_VISIBLE_DEVICES=5;inference de en $step concat.)& \
    (export CUDA_VISIBLE_DEVICES=6;inference en de $step concat.)& \
    (export CUDA_VISIBLE_DEVICES=7;inference en zh $step concat.)& \
    (export CUDA_VISIBLE_DEVICES=4;inference zh en $step concat.)
    wait
done