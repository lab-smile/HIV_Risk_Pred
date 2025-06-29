# Learning on Forecasting HIV Epidemic Based on Individuals' Contact Networks

Improving the diagnosis of HIV is a fundamental objective of the Ending the HIV Epidemic initiative, as it represents the initial step toward treatment and achieving undetectable status, thereby reducing transmission. To attain these objectives effectively, it is crucial to identify the groups most susceptible to HIV, allowing interventions to be tailored to their specific needs.
In this study, we developed a predictive model designed to assess individual HIV risk within a high-risk contact network -- predicting treatment or at-risk -- leveraging surveillance data collected through routine HIV case interviews in Florida. Unique to our analysis, we explored the incorporation of behavioral network information with Graph Neural Networks to enhance the predictive capacity for identifying individuals within the treatment or intervention categories, when compared to models that mainly consider conventional HIV risk factors. Our deployed HIVForecast learning framework achieved 78.2% and 73.9% balanced accuracy in inductive and transductive learning scenarios respectively, outperforming the conventional prediction algorithms that do not leverage the network structure. We then used our framework to further investigate the importance of demographic and behavioral factors in the HIV risk prediction process and the changing trends of patients demographics across years. Our findings provide valuable insights for healthcare practitioners and policymakers in their efforts to combat HIV infection. An early version of this work won the **Best Paper Award** in The 17th International Conference on Health Informatics (HEALTHINF/BIOSTEC), 2024.

## Dependencies
Please check the dependencies.txt.

## Datasets
- We received data extracts from FDOH’s STARS in a fully de-identified format according to the Health Insurance Portability and Accountability Act (HIPAA). For replication purposes, a STARS data request to the [FDOH] can be made according to state, federal regulations and compliance with required ethical and privacy policies, including IRB approval by FDOH and execution of data user agreement. Requests are independently reviewed by FDOH.

## Usage examples
- Train models  
The frameworks of inductive learning task and transductive learning task are included in [dl] and [dl_time] respectively. Both of them support training Graph SAGE, GIN, and HIVForecast models. 

Example commands 
```sh
# train a GIN model
python main.py --model 'gin' --model_num 0 --batch_size 32 --init_lr 0.001 --min_lr 1e-6
```

<!--
## Paper
This repository provides the official implementation of the model in the following paper:

**Learning on Forecasting HIV Epidemic Based on Individuals' Contact Networks**

Chaoyue Sun<sup>1</sup>, Yiyang Liu<sup>2</sup>, Christina Parisi<sup>2</sup>, Rebecca Fisk-Hoffman<sup>2</sup>, Marco Salemi<sup>3,4</sup>, Ruogu Fang<sup>1,4,5,6</sup>, Brandi Danforth<sup>7</sup>, Mattia Prosperi<sup>2,4</sup> and Simone Marini<sup>2,4*</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, Herbert Wertheim College of Engineering, University of Florida, Gainesville, FL, USA<br>
<sup>2</sup> Department of Epidemiology, College of Public Health and Health Professions and College of Medicine, University of Florida, Gainesville, FL, USA<br>
<sup>3</sup> Department of Pathology, Immunology and Laboratory Medicine, College of Medicine, University of Florida, Gainesville, FL, USA<br>
<sup>4</sup> Emerging Pathogens Institute, University of Florida, Gainesville, FL, USA<br>
<sup>5</sup> J. Crayton Pruitt Family Department of Biomedical Engineering, Herbert Wertheim College of Engineering, University of Florida, Gainesville, FL, USA<br>
<sup>6</sup> Center for Cognitive Aging and Memory, McKnight Brain Institute, University of Florida, Gainesville, FL, USA<br>
<sup>7</sup> Florida Department of Health, 4025 Esplanade Way, Tallahassee, FL, USA<br>

## Citation
If you use this code, please cite our papers:
```
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
-->

## Acknowledgement
The authors abide to the Declaration of Helsinki. The study protocol was approved by the University of Florida’s Institutional Review Board (IRB) and by FDOH’s IRB (protocol #IRB201901041 and #2020-069, respectively) as exempt. We received data extracts from FDOH’s STARS in a fully de-identified format according to the Health Insurance Portability and Accountability Act (HIPAA). For replication purposes, a STARS data request to the FDOH can be made according to state, federal regulations and compliance with required ethical and privacy policies ([FDOH]), including IRB approval by FDOH and execution of data user agreement. Requests are independently reviewed by FDOH. We would like to express our gratitude to Colby Cohen and Jared Jashinsky from FDOH for their invaluable assistance in preparing the STARS data for our analysis, for their responsiveness to our inquiries regarding the dataset, and for their instrumental role in facilitating the FDOH internal review and approval process for our manuscript. 

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [FDOH]: <Research@flhealth.gov>
   [dl]: <https://github.com/lab-smile/HIV_Risk_Pred/tree/main/dl>
   [dl_time]: <https://github.com/lab-smile/HIV_Risk_Pred/tree/main/dl_time>
