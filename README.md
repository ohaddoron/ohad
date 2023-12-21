# MMRACS (Multi Modality Risk Assessment with Applications to Cancer Survival)

## Introduction

Welcome to the MMRACS repository, my comprehensive platform showcasing the research on multi-modality risk assessment in
cancer survival prediction. This repository encapsulates the work and findings from my study, offering insights into the
intersection of healthcare analytics, machine learning, and risk assessment strategies.

### Background

Multi-modality Risk Assessment (MRA) represents a holistic approach in data analysis, integrating diverse data types
like text, images, audio, and video. This method provides a more nuanced understanding of risks than traditional
single-modality methods, especially critical in healthcare, finance, and cybersecurity applications.

In this research, we focus on harnessing MRA for cancer survival prediction. Leveraging advanced techniques in machine
learning and deep learning, the study aims to address challenges in integrating data from heterogeneous sources. A
key component of this research is the utilization of the comprehensive TCGA dataset, a rich resource in cancer research,
offering a unique blend of opportunities and challenges due to its complexity and incomplete records.

### The Problem

Conventional risk assessment models often struggle with incomplete or less informative data modalities, a prevalent
challenge in cancer survival prediction. This repository presents novel methodologies developed in my research to
robustly handle data sparsity and incompleteness.

### Objectives

The aim of this repository is to provide a detailed overview of a novel approach for multi-modal risk assessment in
cancer survival. The focus is on efficient handling of missing data, synthesizing knowledge across diverse data
modalities, and introducing a nuanced risk measurement method. The objective is to enhance the predictive accuracy and
practical applicability of risk assessment models in real-world scenarios.

### Contributions

1. **Efficient Handling of Missing Data:** The research presents methods designed to effectively manage incomplete
   datasets, a common issue in healthcare modeling.

2. **Knowledge Distillation Across Modalities:** The study explores how to use information from data-rich modalities to
   improve understanding in data-scarce areas, ensuring adaptability and continual evolution of the system.

3. **Stratified Risk Measurement:** Departing from traditional risk assessment methods, this research introduces a
   multi-tiered approach to risk evaluation, allowing for more granular and practical insights.

This repository contains the datasets, code, and detailed documentation from my research. It serves as a resource for
those interested in the advanced methodologies developed and their applications in cancer survival predictions,
demonstrating a significant impact on patient outcomes and healthcare strategies.

## Reproducing Prior Work

### Data

The data used in this research is sourced from
the [The Cancer Genome Atlas (TCGA)](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga).
The TCGA dataset is a comprehensive resource for cancer research, offering a rich collection of clinical, genomic, and
radiomic data. The data is available for download from the [GDC Data Portal](https://portal.gdc.cancer.gov/).

### Reproducing Prior Art

Along the research, I have gathered and reutilized some of the leading works in the field of cancer survival prediction.
Some of the key papers are listed below along with links to the revised structuring of the code and data is provided in
the following sections:

* [multisurv: Long-term cancer survival prediction using multimodal deep learning](https://www.nature.com/articles/s41598-021-92799-4?proof=t)
    * [Code](https://github.com/ohaddoron/multisurv.git)
* [MultimodalSurvivalPrediction: Pancancer survival prediction using a deep learning architecture with multimodal representation and integration](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad006/6998218)
    * [Code](https://github.com/luisvalesilva/multisurv.git)
* [MultimodalPrognosis: Deep Learning with Multimodal Representation for Pancancer Prognosis Prediction](https://github.com/ohaddoron/MultimodalPrognosis.git)
    * [Code](https://github.com/ohaddoron/MultimodalPrognosis.git)

All the code for prior art presented here was forked from the original repositories and updated to work within a docker
container for easy reproducibility.




