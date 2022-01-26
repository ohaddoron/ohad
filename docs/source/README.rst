This thesis deals with the question of how to handle missing medical data when attempting to predict survival of breast cancer patients.
Patient's data is collected by multiple modalities, ranging from genomic, proteomic, metabolmic radiomic and pathomic data. Each different modality provides a different outlook on the patient and may contain valuable information in determining the survival of the patient and the recommended course of treatment. In this work, I attempt to provide a generic framework for handling missing data in the context of these modalities.

The project is based on data taken from the `GDC Portal <https://portal.gdc.cancer.gov/exploration?facetTab=cases&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-BRCA%22%5D%7D%7D%5D%7D>`_ that was inserted into a custom MongoDB.


