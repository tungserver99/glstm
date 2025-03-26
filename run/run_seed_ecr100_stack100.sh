for seed in {1..10}; do
    for aug_coef in 1; do
        for prior_var in 0.1; do
            for weight_loss_ECR in 100; do
                echo "Running with seed=$seed, aug_coef=$aug_coef, prior_var=$prior_var, weight_loss_ECR=$weight_loss_ECR"
                python run.py --wandb_prj glolo-knn --model GloCOM --global_dir global_knn_30 --num_topics 100 --device cuda:0 \
                              --seed $seed --aug_coef $aug_coef --prior_var $prior_var \
                              --weight_loss_ECR $weight_loss_ECR --data_dir data/StackOverflow
            done
        done
    done
done
