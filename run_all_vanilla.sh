scenes=('fern'  'flower'  'fortress'  'horns'  'leaves'  'orchids'  'room'  'trex')
method=vanilla
number_of_views_list=(3 6)

for n_views in "${number_of_views_list[@]}"; do
    for scene in "${scenes[@]}"; do
        python train.py --config configs/flower_fourier.txt \
                        --number_of_views 3\
                        --expname "$scene"_"$n_views"\
                        --basedir "./log/$n_views/$method"
    done
done
