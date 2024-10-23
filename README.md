[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

# Task 1.5
## Simple

![Alt text](images/simple.png)
![Alt text](images/simple-loss.png)
Epoch: 0/500, loss: 32.972865, correct: 0
Epoch: 10/500, loss: 32.873761, correct: 31
Epoch: 20/500, loss: 32.756923, correct: 31
Epoch: 30/500, loss: 32.411792, correct: 31
Epoch: 40/500, loss: 32.203366, correct: 31
Epoch: 50/500, loss: 32.368387, correct: 31
Epoch: 60/500, loss: 32.066924, correct: 31
Epoch: 70/500, loss: 31.719267, correct: 31
Epoch: 80/500, loss: 31.046995, correct: 31
Epoch: 90/500, loss: 30.769427, correct: 31
Epoch: 100/500, loss: 30.057721, correct: 31
Epoch: 110/500, loss: 29.257909, correct: 32
Epoch: 120/500, loss: 28.323384, correct: 33
Epoch: 130/500, loss: 27.750909, correct: 35
Epoch: 140/500, loss: 26.554887, correct: 38
Epoch: 150/500, loss: 25.525652, correct: 39
Epoch: 160/500, loss: 24.470665, correct: 39
Epoch: 170/500, loss: 23.313216, correct: 40
Epoch: 180/500, loss: 22.532699, correct: 41
Epoch: 190/500, loss: 21.355581, correct: 42
Epoch: 200/500, loss: 20.180060, correct: 42
Epoch: 210/500, loss: 19.206773, correct: 43
Epoch: 220/500, loss: 17.774435, correct: 44
Epoch: 230/500, loss: 17.196362, correct: 44
Epoch: 240/500, loss: 16.087470, correct: 44
Epoch: 250/500, loss: 15.054579, correct: 45
Epoch: 260/500, loss: 14.747758, correct: 45
Epoch: 270/500, loss: 13.898998, correct: 45
Epoch: 280/500, loss: 12.884828, correct: 45
Epoch: 290/500, loss: 12.384548, correct: 46
Epoch: 300/500, loss: 12.145697, correct: 46
Epoch: 310/500, loss: 11.154297, correct: 47
Epoch: 320/500, loss: 10.977136, correct: 47
Epoch: 330/500, loss: 10.623393, correct: 47
Epoch: 340/500, loss: 10.236800, correct: 47
Epoch: 350/500, loss: 9.805501, correct: 47
Epoch: 360/500, loss: 9.545318, correct: 47
Epoch: 370/500, loss: 9.075397, correct: 47
Epoch: 380/500, loss: 8.568750, correct: 47
Epoch: 390/500, loss: 8.455025, correct: 47
Epoch: 400/500, loss: 8.174800, correct: 47
Epoch: 410/500, loss: 7.818320, correct: 49
Epoch: 420/500, loss: 7.670063, correct: 49
Epoch: 430/500, loss: 7.798518, correct: 49
Epoch: 440/500, loss: 7.432025, correct: 49
Epoch: 450/500, loss: 7.148146, correct: 49
Epoch: 460/500, loss: 6.812164, correct: 49
Epoch: 470/500, loss: 6.659945, correct: 49
Epoch: 480/500, loss: 6.422327, correct: 49
Epoch: 490/500, loss: 6.449827, correct: 49
Epoch: 500/500, loss: 6.555439, correct: 49

---
## Diag
![Alt text](images/diag.png)
![Alt text](images/diag-loss.png)

Epoch: 10/500, loss: 18.334820, correct: 45
Epoch: 20/500, loss: 15.263506, correct: 45
Epoch: 30/500, loss: 14.664088, correct: 45
Epoch: 40/500, loss: 14.467435, correct: 45
Epoch: 50/500, loss: 13.887840, correct: 45
Epoch: 60/500, loss: 13.444743, correct: 45
Epoch: 70/500, loss: 13.268555, correct: 45
Epoch: 80/500, loss: 12.840565, correct: 45
Epoch: 90/500, loss: 12.598231, correct: 45
Epoch: 100/500, loss: 12.509115, correct: 45
Epoch: 110/500, loss: 11.843112, correct: 45
Epoch: 120/500, loss: 11.486038, correct: 45
Epoch: 130/500, loss: 11.517359, correct: 45
Epoch: 140/500, loss: 10.917729, correct: 45
Epoch: 150/500, loss: 10.587992, correct: 45
Epoch: 160/500, loss: 10.166471, correct: 45
Epoch: 170/500, loss: 10.072260, correct: 45
Epoch: 180/500, loss: 9.409280, correct: 45
Epoch: 190/500, loss: 9.125054, correct: 45
Epoch: 200/500, loss: 8.763983, correct: 45
Epoch: 210/500, loss: 8.647193, correct: 45
Epoch: 220/500, loss: 8.215841, correct: 45
Epoch: 230/500, loss: 7.597145, correct: 45
Epoch: 240/500, loss: 7.284387, correct: 45
Epoch: 250/500, loss: 7.173514, correct: 46
Epoch: 260/500, loss: 6.997629, correct: 46
Epoch: 270/500, loss: 6.445040, correct: 48
Epoch: 280/500, loss: 6.034671, correct: 48
Epoch: 290/500, loss: 5.900196, correct: 48
Epoch: 300/500, loss: 5.841729, correct: 48
Epoch: 310/500, loss: 5.411213, correct: 49
Epoch: 320/500, loss: 5.420519, correct: 49
Epoch: 330/500, loss: 4.815486, correct: 49
Epoch: 340/500, loss: 4.927169, correct: 49
Epoch: 350/500, loss: 4.538738, correct: 49
Epoch: 360/500, loss: 4.661285, correct: 49
Epoch: 370/500, loss: 4.657767, correct: 49
Epoch: 380/500, loss: 4.236319, correct: 49
Epoch: 390/500, loss: 4.142453, correct: 49
Epoch: 400/500, loss: 4.267243, correct: 49
Epoch: 410/500, loss: 4.137563, correct: 49
Epoch: 420/500, loss: 4.129586, correct: 49
Epoch: 430/500, loss: 4.091234, correct: 49
Epoch: 440/500, loss: 3.954416, correct: 49
Epoch: 450/500, loss: 4.031296, correct: 49
Epoch: 460/500, loss: 3.848636, correct: 49
Epoch: 470/500, loss: 3.855202, correct: 49
Epoch: 480/500, loss: 3.855232, correct: 49
Epoch: 490/500, loss: 3.712296, correct: 49
Epoch: 500/500, loss: 3.385273, correct: 49


---
## Split
![Alt text](images/split.png)
![Alt text](images/split-loss.png)

Epoch: 10/500, loss: 33.688151, correct: 29
Epoch: 20/500, loss: 33.286750, correct: 30
Epoch: 30/500, loss: 32.774536, correct: 33
Epoch: 40/500, loss: 32.758493, correct: 34
Epoch: 50/500, loss: 32.201312, correct: 36
Epoch: 60/500, loss: 32.236078, correct: 37
Epoch: 70/500, loss: 31.853243, correct: 37
Epoch: 80/500, loss: 31.860696, correct: 36
Epoch: 90/500, loss: 31.474724, correct: 36
Epoch: 100/500, loss: 31.328744, correct: 36
Epoch: 110/500, loss: 30.992593, correct: 36
Epoch: 120/500, loss: 30.599081, correct: 35
Epoch: 130/500, loss: 30.081814, correct: 35
Epoch: 140/500, loss: 29.797277, correct: 35
Epoch: 150/500, loss: 29.786116, correct: 35
Epoch: 160/500, loss: 29.410429, correct: 35
Epoch: 170/500, loss: 29.101527, correct: 36
Epoch: 180/500, loss: 28.601691, correct: 36
Epoch: 190/500, loss: 28.323351, correct: 36
Epoch: 200/500, loss: 28.083963, correct: 36
Epoch: 210/500, loss: 27.405163, correct: 36
Epoch: 220/500, loss: 27.265241, correct: 37
Epoch: 230/500, loss: 26.637113, correct: 38
Epoch: 240/500, loss: 26.268316, correct: 38
Epoch: 250/500, loss: 25.397507, correct: 39
Epoch: 260/500, loss: 25.192053, correct: 40
Epoch: 270/500, loss: 24.395869, correct: 41
Epoch: 280/500, loss: 23.645937, correct: 44
Epoch: 290/500, loss: 23.438372, correct: 44
Epoch: 300/500, loss: 22.621045, correct: 46
Epoch: 310/500, loss: 21.652514, correct: 46
Epoch: 320/500, loss: 21.308657, correct: 46
Epoch: 330/500, loss: 20.663760, correct: 46
Epoch: 340/500, loss: 19.803961, correct: 46
Epoch: 350/500, loss: 19.201525, correct: 46
Epoch: 360/500, loss: 18.438380, correct: 47
Epoch: 370/500, loss: 17.754417, correct: 48
Epoch: 380/500, loss: 16.875735, correct: 48
Epoch: 390/500, loss: 16.385707, correct: 49
Epoch: 400/500, loss: 15.715484, correct: 50
Epoch: 410/500, loss: 14.904487, correct: 50
Epoch: 420/500, loss: 14.253530, correct: 50
Epoch: 430/500, loss: 13.713700, correct: 50
Epoch: 440/500, loss: 13.049208, correct: 50
Epoch: 450/500, loss: 12.703695, correct: 50
Epoch: 460/500, loss: 11.851971, correct: 50
Epoch: 470/500, loss: 11.342098, correct: 50
Epoch: 480/500, loss: 10.905041, correct: 50
Epoch: 490/500, loss: 10.558782, correct: 50
Epoch: 500/500, loss: 10.214309, correct: 50


---
## Xor
![Alt text](images/xor.png)
![Alt text](images/xor-loss.png)

Epoch: 10/500, loss: 31.755551, correct: 39
Epoch: 20/500, loss: 29.615886, correct: 39
Epoch: 30/500, loss: 27.050288, correct: 42
Epoch: 40/500, loss: 26.907972, correct: 38
Epoch: 50/500, loss: 25.004914, correct: 38
Epoch: 60/500, loss: 23.745144, correct: 38
Epoch: 70/500, loss: 22.085623, correct: 40
Epoch: 80/500, loss: 22.634336, correct: 38
Epoch: 90/500, loss: 18.091819, correct: 43
Epoch: 100/500, loss: 18.927711, correct: 43
Epoch: 110/500, loss: 14.225782, correct: 44
Epoch: 120/500, loss: 17.646423, correct: 44
Epoch: 130/500, loss: 13.447640, correct: 44
Epoch: 140/500, loss: 13.466825, correct: 43
Epoch: 150/500, loss: 12.759727, correct: 45
Epoch: 160/500, loss: 11.926453, correct: 44
Epoch: 170/500, loss: 11.565316, correct: 45
Epoch: 180/500, loss: 12.276330, correct: 44
Epoch: 190/500, loss: 11.317784, correct: 43
Epoch: 200/500, loss: 8.624023, correct: 47
Epoch: 210/500, loss: 7.698529, correct: 47
Epoch: 220/500, loss: 7.828972, correct: 47
Epoch: 230/500, loss: 7.551643, correct: 47
Epoch: 240/500, loss: 5.671300, correct: 49
Epoch: 250/500, loss: 11.618703, correct: 45
Epoch: 260/500, loss: 5.690076, correct: 48
Epoch: 270/500, loss: 5.070931, correct: 49
Epoch: 280/500, loss: 4.773259, correct: 49
Epoch: 290/500, loss: 5.330103, correct: 49
Epoch: 300/500, loss: 4.771746, correct: 49
Epoch: 310/500, loss: 3.936281, correct: 49
Epoch: 320/500, loss: 4.015620, correct: 49
Epoch: 330/500, loss: 4.693532, correct: 49
Epoch: 340/500, loss: 3.694234, correct: 49
Epoch: 350/500, loss: 3.745528, correct: 49
Epoch: 360/500, loss: 3.615878, correct: 49
Epoch: 370/500, loss: 3.563155, correct: 49
Epoch: 380/500, loss: 3.900066, correct: 49
Epoch: 390/500, loss: 3.325746, correct: 49
Epoch: 400/500, loss: 3.085689, correct: 49
Epoch: 410/500, loss: 3.072115, correct: 49
Epoch: 420/500, loss: 4.019121, correct: 49
Epoch: 430/500, loss: 3.580987, correct: 49
Epoch: 440/500, loss: 2.969077, correct: 49
Epoch: 450/500, loss: 2.912283, correct: 49
Epoch: 460/500, loss: 2.982386, correct: 49
Epoch: 470/500, loss: 2.953817, correct: 49
Epoch: 480/500, loss: 3.430603, correct: 49
Epoch: 490/500, loss: 3.365671, correct: 49
Epoch: 500/500, loss: 3.064287, correct: 49

