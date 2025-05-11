# Market Regime Detection Using Gaussian Hidden Markov Models

**Author:** Tumelo Tshwanelo Mkwambe  
**Institution:** School of Computer Science and Applied Mathematics, University of the Witwatersrand, Johannesburg, South Africa  
**Email:** 2446873@students.wits.ac.za  
**Full Report:** [Full project report](/report.pdf)

## Abstract

A **market regime** refers to a period in financial markets that exhibits consistent behaviors or patterns. Typical regimes include:
- **Bull markets** (optimism, rising prices)
- **Bear markets** (pessimism, falling prices)
- **Volatility** (uncertainty, risk)

This project applies **Gaussian Hidden Markov Models (HMMs)** to detect and characterize these regimes in the **South African Top 40 Index (JTOPI)**.

---

## 1. Introduction

The **South African Top 40 Index (JTOPI)** is a capitalization-weighted index of the top 40 companies on the Johannesburg Stock Exchange.  
**Hidden Markov Models (HMMs)** are used to model this index's time series data by assuming the market switches between hidden states with distinct return and volatility patterns.

---

## 2. Problem Formulation

To support risk assessment and strategy development, quantitative analysts need methods to infer **market sentiment**.  
This study addresses the challenge of deducing such sentiment from **historical and real-time data** using HMMs.

---

## 3. Data and Analysis

### A. Data
- Source: [Investing.com](https://www.investing.com)
- Timeframe: `2015-01-01` to `2025-01-01`
- Data Fields: `Open`, `Close`, `Volume`
- Preprocessing: Cleaned and null values handled.

### B. Analysis
- **Log returns** were used instead of raw prices due to their stationarity and suitability for Gaussian HMMs.
- Statistical summary:

| Statistic | Value |
|----------|-------|
| Minimum | -0.1045 |
| Maximum | 0.0906 |
| Mean    | 0.0002 |
| Variance | 0.00014 |
| Skewness | -0.308 |
| Kurtosis | 7.135 |

---

## 4. Hidden Markov Models

A **Hidden Markov Model (HMM)** consists of:
- States: `Q = {q0, q1, ..., qN−1}`
- Transition matrix `A ∈ ℝⁿˣⁿ`
- Emission probabilities `B ∈ ℝⁿˣᵐ`
- Initial probabilities `π ∈ ℝⁿ`

### Core Problems Solved by HMMs
1. **Likelihood** - Compute `P(O | λ)` using the **Forward Algorithm**
2. **Decoding** - Find the optimal state sequence using **Viterbi Algorithm**
3. **Learning** - Adjust model parameters using **Baum-Welch Algorithm**

---

## 5. Python Libraries
`numpy`
`pandas`
`scipy.stats`
`matplotlib.pyplot`
`matplotlib.dates`
`matplotlib.cm`
`seaborn`
`pickle`

---

## 6. Conclusion

This project demonstrates how **Gaussian Hidden Markov Models** can effectively capture and characterize market regimes. The approach enables:
- Detection of **underlying state transitions**
- Better **risk management** and **investment decision-making**

---
