
# 12 Leads ECG Signal Reconstruction from DII Lead

12 Leads ECG is essential for examining cardio status. However, most commercial-grade products are unable to capture full 12 Leads ECG signal. And so, Reconstructed 12 Leads ECG Signal might be a better solution for identifying arrhythmia for data collected by smart devices. Here I bulid a Deep Learning model with U-Net Architecture for this specific task.

Here are reconstructed ECG charts of this model.
![ID_140](https://user-images.githubusercontent.com/76748651/215316913-71177605-43cb-460e-aac3-32e2fd8b34b5.png)
![ID_547](https://user-images.githubusercontent.com/76748651/215317048-f3b76319-6def-4b8e-9f10-f0c1142329c8.png)

Chart of Losses on each epochs
![U-Net_Loss_Result](https://user-images.githubusercontent.com/76748651/215317076-23cef54e-a752-441d-b66e-844457b600e0.png)

# Dataset Description 

I built this model with Code-15% dataset. You can find the description and original data from following link.
https://zenodo.org/record/4916206

You can also download cleaned data for this particular project via following Link:
https://1drv.ms/u/s!ArQCikHAsFj6oPJoQY7h9rIggjoaug?e=YjNprR

# Reference:

1. Learning to See in the Dark: https://arxiv.org/abs/1805.01934

2. I am sorry that I lost the original source of my U-Net Implementation. Please let me know if you are the author.
