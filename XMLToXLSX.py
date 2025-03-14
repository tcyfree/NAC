import xml.etree.ElementTree as ET
import pandas as pd
import os

# 定义命名空间（确保正确解析 XML 标签）
ns = {
    'brca': 'http://tcga.nci/bcr/xml/clinical/brca/2.7',
    'shared': 'http://tcga.nci/bcr/xml/shared/2.7',
    'clin_shared': 'http://tcga.nci/bcr/xml/clinical/shared/2.7',
    'brca_shared': 'http://tcga.nci/bcr/xml/clinical/brca/shared/2.7',
    'shared_stage': 'http://tcga.nci/bcr/xml/clinical/shared/stage/2.7',
    'rx': 'http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7'
}

# 用于保存所有患者数据的列表
patients = []

# 遍历文件夹及其子文件夹
for root_dir, dirs, files in os.walk('./data/TCGA/Clinical'):  # 请根据你的文件夹路径修改
    for file in files:
        if file.endswith('.xml'):
            xml_file_path = os.path.join(root_dir, file)
            print(f'正在处理文件: {xml_file_path}')
            
            # 解析 XML 文件
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # 遍历患者数据
            for patient in root.findall('.//brca:patient', namespaces=ns):
                patient_data = {}

                # 提取临床信息
                patient_data['Patient_ID'] = patient.findtext('.//shared:bcr_patient_barcode', namespaces=ns)
                patient_data['Gender'] = patient.findtext('.//shared:gender', namespaces=ns)
                patient_data['Age'] = patient.findtext('.//clin_shared:age_at_initial_pathologic_diagnosis', namespaces=ns)
                patient_data['Tumor Stage'] = patient.findtext('.//shared_stage:pathologic_stage', namespaces=ns)
                patient_data['ER Status'] = patient.findtext('.//brca_shared:breast_carcinoma_estrogen_receptor_status', namespaces=ns)
                patient_data['PR Status'] = patient.findtext('.//brca_shared:breast_carcinoma_progesterone_receptor_status', namespaces=ns)
                patient_data['HER2 Status'] = patient.findtext('.//brca_shared:lab_proc_her2_neu_immunohistochemistry_receptor_status', namespaces=ns)
                patient_data['Vital Status'] = patient.findtext('.//clin_shared:vital_status', namespaces=ns)
                patient_data['Last Followup (days)'] = patient.findtext('.//clin_shared:days_to_last_followup', namespaces=ns)
                patient_data['Neoadjuvant Therapy'] = patient.findtext('.//shared:history_of_neoadjuvant_treatment', namespaces=ns)

                # 额外提取字段
                patient_data['Histologic Type'] = patient.findtext('.//shared:histological_type', namespaces=ns)
                patient_data['Lymph Nodes Examined'] = patient.findtext('.//clin_shared:lymph_node_examined_count', namespaces=ns)
                patient_data['Menopause Status'] = patient.findtext('.//clin_shared:menopause_status', namespaces=ns)
                patient_data['Margin Status'] = patient.findtext('.//clin_shared:margin_status', namespaces=ns)
                patient_data['Surgical Procedure'] = patient.findtext('.//brca:breast_carcinoma_surgical_procedure_name', namespaces=ns)
                patient_data['Radiation Therapy'] = patient.findtext('.//clin_shared:radiation_therapy', namespaces=ns)
                patient_data['Chemotherapy'] = patient.findtext('.//rx:therapy_type', namespaces=ns)
                patient_data['Prescribed Dose'] = patient.findtext('.//rx:prescribed_dose', namespaces=ns)
                patient_data['Drug Name'] = patient.findtext('.//rx:drug_name', namespaces=ns)
                patient_data['Therapy Ongoing'] = patient.findtext('.//rx:therapy_ongoing', namespaces=ns)

                # 追加到列表
                patients.append(patient_data)

# 转换为 Pandas DataFrame
df = pd.DataFrame(patients)

# 保存为 CSV 文件
df.to_csv('./data/TCGA/combined_clinical_data.csv', index=False)

print("转换完成！所有数据已保存到 'combined_clinical_data.csv'")
