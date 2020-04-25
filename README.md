## Code for math word problems solver using Bi-LSTM with equation normalization
----
We rewrote the code, and at present we only finished two normalization methods in this code (test acc: 65.5%), cause I haven't found previous preprocessing code, I will complement another normalization method as soon as possible.(but we provide the original norm files, so you can use directly) Thus, I directly used the previous templates with EN to do the entire experiments (test acc: ~67%).

This code requires python 3.5, pytorch 0.4 and some common python tools.

Code for data processing is in the data_process.ipynb

For SNI, we directly use the results from the paper "Deep Neural Solver for Math Word Problems". You can find relevant data from ./data/sni_dict.json

```sh
sh ./script/exe_post.sh model_dir
```

The training accuracy, and test accuracy will be printed. (you can randomly sample 1000 problems as validation set to help you tune your hyperparameters, and then train the model based on all training problems with the hyperparameters you choose)

## References
----
- Lei Wang, Yan Wang, Deng Cai, Dongxiang Zhang, Xiaojiang Liu, "Translating a Math Word Problem to an Expression Tree", https://arxiv.org/abs/1811.05632
[![Run on Repl.it](https://repl.it/badge/github/SumbeeLei/Math_EN)](https://repl.it/github/SumbeeLei/Math_EN)