# Intraday Market Activity Clustering: Insights from Macroeconomic Signals 📊💡

## Abstract 📝

Financial markets are complex, influenced by various participants and external
factors such as geopolitical events and economic indicators. By using high-frequency intraday
data, we aim to uncover patterns in market behavior that can enhance trading strategies. Our
approach incorporates both likelihood-based PGA and graph-based clustering techniques to
analyze the dynamics of market states across different time scales. The results demonstrate
the potential of these models in identifying distinct market regimes, particularly in distinguish-
ing between different macroeconomic periods. This work lays the foundation for more effective
predictive models in financial market analysis and decision-making, with the potential to incor-
porate external macroeconomic factors to improve predictive accuracy and the interpretation of
market behavior.

---

## Getting Started 🚀

Welcome to the **Intraday Market Activity Clustering** project! This repository contains the code and resources to help you reproduce the results from our research paper and explore clustering techniques applied to market data using macroeconomic signals.

### Prerequisites 🛠️

Before you start, you'll need to install a few dependencies:

You can install all required dependencies by running:

`pip install -r requirements.txt`

### Setup 🏁

Clone the repository:

`git clone https://github.com/yourusername/clustering-research.git`  
`cd clustering-research`

Make sure your environment is set up correctly and all dependencies are installed. Now you are ready to run the experiments and start exploring!

---

## Where to Find the Classes 🗂️

The core functionality of the clustering methods is organized into Python classes. You can find these in the `src/` directory:

- **Clustering:** Contains the implementations of various clustering methods.


You can explore each of these classes in their respective files.

---

## Run Experiments 🔬

NOTE: Data is available at https://drive.switch.ch/index.php/s/0X3Je6DauQRzD2r?path=%2FCAC40

To reproduce the experimental results from our research paper, simply run the `run_experiments.py` script. This script will process the data, apply the clustering algorithms, and output the results in the same way they appear in the paper.

### Usage 📥

`python run_experiments.py`

---

## Examples 💻

We’ve provided a set of Jupyter notebooks and scripts in the `examples/` directory to give you a hands-on guide for using the clustering methods in various real-world scenarios.

### Explore the following examples:

- **Example 1:** Clustering Intraday Market With Louvain
- **Example 2:** Clustering Intraday Market With PGA
- **Example 3:** Clustering Intraday Market With Random (Test purposes)

These notebooks will help you get a deeper understanding of how to apply the clustering methods to your own data, tweak the settings, and interpret the results.
