# HIV_Risk_Pred

This is a pytorch implementation of our research:  
Learning on forecasting HIV epidemic based on patients' contact networks, accepted at 17th International Conference on Health Informatics (HEALTHINF/BIOSTEC).
If you find our work useful in your research or publication, please cite our work:
```sh
@conference{healthinf24,
author={Chaoyue Sun. and Yiyang Liu. and Christina Parisi. and Rebecca Fisk{-}Hoffman. and Marco Salemi. and Ruogu Fang. and Brandi Danforth. and Mattia Prosperi. and Simone Marini.},
title={Learning on Forecasting HIV Epidemic Based on Individuals' Contact Networks},
booktitle={Proceedings of the 17th International Joint Conference on Biomedical Engineering Systems and Technologies - HEALTHINF},
year={2024},
pages={103-111},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012375400003657},
isbn={978-989-758-688-0},
issn={2184-4305},
}
```

## Dependencies
Please check the dependencies.txt.

## Datasets
- We received data extracts from FDOH’s STARS in a fully de-identified format according to the Health Insurance Portability and Accountability Act (HIPAA). For replication purposes, a STARS data request to the [FDOH] can be made according to state, federal regulations and compliance with required ethical and privacy policies, including IRB approval by FDOH and execution of data user agreement. Requests are independently reviewed by FDOH.

## Usage examples
- Train models  
The frameworks of inductive learning task and transductive learning task are included in [dl] and [dl_time] respectively. Both of them support training GCN and GIN models. 

Example commands 
```sh
# train a GIN model
python main.py --model 'gin' --model_num 0 --batch_size 32 --init_lr 0.001 --min_lr 1e-6
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [FDOH]: <Research@flhealth.gov>
   [dl]: <https://github.com/lab-smile/HIV_Risk_Pred/tree/main/dl>
   [dl_time]: <https://github.com/lab-smile/HIV_Risk_Pred/tree/main/dl_time>
