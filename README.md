Welcome to my project :)

# Dependencies

If you want to re-run this code, please first follow the instructions on [SAT](https://github.com/zyang-ur/SAT) and [referit3D](https://github.com/referit3d/referit3d/). In addition to downloading the dataset. NOTE: `referit3d/data` and `referit3d/external_tools` are not supposed to be empty folders! I emptied them for convenience as they're heavy. Please clone them from the two links above.

# Running the code

Once the environment is ready, you can run the scrip on ibex using `sbatch bin/ibex_first.sh` for the best model using original SAT baseline and `sbatch bin/ibex_second.sh` for the model not using object correspondence loss originally presented in SAT, or simply run this command on terminal after activating the environment (can also be found in `bin` under `bin/terminal_script_best.sh` and `bin/terminal_script_second.sh`)

```bash
# best model
python referit3d/scripts/train_referit3d.py -scannet-file referit3d/data/scannet/save_dir/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl -referit3D-file referit3d/data/language/nr3d/csv/nr3d.csv --log-dir results/BEST --n-workers 6 --patience 100 --max-train-epochs 100 --init-lr 1e-4 --augment-with-sr3d referit3d/data/language/sr3d/csv/sr3d_train.csv  --unit-sphere-norm True --feat2d clsvecROI --context_2d unaligned --mmt_mask train2d --warmup --batch-size 32  --transformer --model mmt_referIt3DNet --margin 1 --ce 1 --triplet 0 --contrastive 1 --ce2 1 --triplet2 0 --contrastive2 0

# second best model (no correspondence loss)
python referit3d/scripts/train_referit3d.py -scannet-file referit3d/data/scannet/save_dir/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl -referit3D-file referit3d/data/language/nr3d/csv/nr3d.csv --log-dir results/SECOND --n-workers 6   --augment-with-sr3d referit3d/data/language/sr3d/csv/sr3d_train.csv --margin 1 --ce 1 --triplet 0 --contrastive 1 --ce2 1 --triplet2 0 --contrastive2 1
```



# Edited Files (contribution)

For easier code navigation, the edited files are:

* `model/losses.py`: a new file presenting the loss function class
* `in_out/pt_datasets/listening_dataset.py`: changed the sampling strategy.
* `model/referit3d_net_utils.py`: the functions were changed to make them compatible with the new code
* `scripts/train_referit3d.py`: changed the main script's criterion to be the new loss function, instead of cross entropy
* `in_out/arguments.py`: added the new arguments for the custom loss. This includes the margin, the 3 weights for the 3D loss function and the 3 weights for the 2D loss function.

# Results

Since each run takes 5(!) days, I included all results files. I did not upload the actual models because each one is ~1GB, but I can provide them upon request. All results are in the `results` folder, divided into folder by the experiment type. For each experiment, I included the configuration and log files



# Acknowledgement

This project is part of KAUST's CS326 course. The project is built on top of referit3d, SAT and uses scannet dataset.



# References

1. Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed
   Elhoseiny, and Leonidas J. Guibas. ReferIt3D: Neural lis-
   teners for fine-grained 3d object identification in real-world
   scenes. In 16th European Conference on Computer Vision
   (ECCV), 2020. 
2. Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner.
   Scanrefer: 3d object localization in rgb-d scans using natu-
   ral language. 16th European Conference on Computer Vision
   (ECCV), 2020. 
3. S. Chopra, R. Hadsell, and Y. LeCun. Learning a similar-
   ity metric discriminatively, with application to face verifica-
   tion. In 2005 IEEE Computer Society Conference on Com-
   puter Vision and Pattern Recognition (CVPR’05), volume 1,
   pages 539–546 vol. 1, 2005. 
4. Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber,
   Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-
   annotated 3d reconstructions of indoor scenes, 2017. 
5. Florian Schroff, Dmitry Kalenichenko, and James Philbin.
   Facenet: A unified embedding for face recognition and clus-
   tering. In Proceedings of the IEEE Conference on Computer
   Vision and Pattern Recognition (CVPR), June 2015. 
6. Naoto Usuyama, Natalia Larios Delgado, Amanda K Hall,
   and Jessica Lundin. epillid dataset: A low-shot fine-grained
   benchmark for pill identification. In Proceedings of the
   IEEE Conference on Computer Vision and Pattern Recogni-
   tion Workshops, 2020.
7. Zhengyuan Yang, Songyang Zhang, Liwei Wang, and Jiebo
   Luo. Sat: 2d semantics assisted training for 3d visual ground-
   ing. In ICCV, 2021.



Thank you for reading thus far :D special thanks to Prof Mohamed and the TAs Jun Chen and Ivan Skorokhodov for helping me throughout this project. 