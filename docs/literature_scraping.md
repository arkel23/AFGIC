# Literature Survey
The original paper list was extracted using [this Colab script](https://colab.research.google.com/drive/1lyKg7tWZBGxNAJ2mVf-6u_N_d8_aIRHp?usp=sharing) for searching arXiv based on keywords (404 matches), then searching the bibliography recursively for other papers that also contain keywords using SemanticScholar API (655 matches).

CSV file of two-level recursive search results on [Drive](https://drive.google.com/file/d/1MmoG4n76cG__tvizL7GkTEUg_s3kPAtt/view?usp=sharing).

Manual filtering based on publication venue prestige and citations [spreadsheet](https://docs.google.com/spreadsheets/d/1ML4kY2wwxBDDhN6EwxuF5SalJyBzuqYC/edit?usp=sharing&ouid=106900472666190121297&rtpof=true&sd=true).

Filtering criteria based on citations and publication venue prestige:
- Any paper from any of the followings regardless of number of citations: IEEE and ACM Transactions, IEEE SP Letters, CVPR, ICCV, ECCV, BMVC, WACV, MM, AAAI, IJCAI, NeurIPS, ICLR, ICML.
- At least 100 citations for papers from 2011 to 2019.
- At least 50 citations for papers from 2020.
- At least 1 citation for papers from 2021.
- All papers from 2022 and 2023.

Filtering based on topic:
- Remove fine-grained entity recognition, fine-grained sentiment analysis and other NLP tasks.
- Remove fine-grained activity and gesture recognition (video).
- Remove fine-grained vision tasks but either using different modalities (RGB-D, 3D, multimodal video or audio datasets) or other learning paradigms (meta-learning / few-shot, open-set, noisy data, web data, active).
- Tag datasets and surveys with yellow on the paper title.
