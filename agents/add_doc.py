from memory import BioClinicalMemoryAgent

EXISTING_JSON = "sample_bioclinical_memory.json"
EXISTING_FAISS = "sample_bioclinical_memory.faiss"

def main():
    agent = BioClinicalMemoryAgent()

    # 1) Load existing memory so nothing is lost
    ok = agent.load_memory(EXISTING_JSON)
    if not ok:
        raise RuntimeError(f"Failed to load {EXISTING_JSON}")

    try:
        stats = agent.get_stats()
        print(f"Loaded: {stats.get('total_documents')} docs, {stats.get('total_chunks')} chunks")
    except:
        print("Loaded existing memory.")

    # Flu case (Patient ID: 1137)
    agent.add_document(
        """25-year-old male presents with cough, fever, and body ache for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.2°C, heart rate 90 bpm, blood pressure 161/81, oxygen saturation 93%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 25, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'fever', 'body_ache', 'high_fever'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.2, 'heart_rate': 90, 'blood_pressure': '161/81', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 1137}
    )

    # Flu case (Patient ID: 1870)
    agent.add_document(
        """24-year-old male presents with fever, headache, and runny nose for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 38.5°C, heart rate 80 bpm, blood pressure 129/119, oxygen saturation 99%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 24, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fever', 'headache', 'runny_nose', 'high_fever', 'normal_oxygenation'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 38.5, 'heart_rate': 80, 'blood_pressure': '129/119', 'oxygen_saturation': 99}, 'setting': 'inpatient', 'patient_id': 1870}
    )

    # Flu case (Patient ID: 741)
    agent.add_document(
        """34-year-old female presents with body ache, fever, and headache for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.3°C, heart rate 90 bpm, blood pressure 123/103, oxygen saturation 92%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 34, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'fever', 'headache', 'high_fever', 'hypoxemia'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.3, 'heart_rate': 90, 'blood_pressure': '123/103', 'oxygen_saturation': 92}, 'setting': 'inpatient', 'patient_id': 741}
    )

    # Flu case (Patient ID: 1556)
    agent.add_document(
        """18-year-old female presents with runny nose, shortness of breath, and fever for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.5°C, heart rate 70 bpm, blood pressure 154/84, oxygen saturation 90%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 18, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['runny_nose', 'shortness_of_breath', 'fever', 'high_fever', 'hypoxemia'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.5, 'heart_rate': 70, 'blood_pressure': '154/84', 'oxygen_saturation': 90}, 'setting': 'inpatient', 'patient_id': 1556}
    )

    # Flu case (Patient ID: 145)
    agent.add_document(
        """46-year-old male presents with fatigue, fever, and shortness of breath for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.3°C, heart rate 72 bpm, blood pressure 97/102, oxygen saturation 93%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 46, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'shortness_of_breath', 'high_fever'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.3, 'heart_rate': 72, 'blood_pressure': '97/102', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 145}
    )

    # Flu case (Patient ID: 1752)
    agent.add_document(
        """60-year-old male presents with sore throat, fever, and body ache for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 38.4°C, heart rate 116 bpm, blood pressure 119/111, oxygen saturation 95%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 60, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['sore_throat', 'fever', 'body_ache', 'low_grade_fever', 'tachycardia'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 38.4, 'heart_rate': 116, 'blood_pressure': '119/111', 'oxygen_saturation': 95}, 'setting': 'inpatient', 'patient_id': 1752}
    )

    # Flu case (Patient ID: 1617)
    agent.add_document(
        """49-year-old female presents with fever, runny nose, and body ache for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 38.5°C, heart rate 72 bpm, blood pressure 165/62, oxygen saturation 91%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 49, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fever', 'runny_nose', 'body_ache', 'high_fever', 'hypoxemia'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 38.5, 'heart_rate': 72, 'blood_pressure': '165/62', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1617}
    )

    # Flu case (Patient ID: 365)
    agent.add_document(
        """50-year-old female presents with fatigue, fever, and headache for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.4°C, heart rate 92 bpm, blood pressure 115/98, oxygen saturation 98%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 50, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'headache', 'high_fever', 'normal_oxygenation'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.4, 'heart_rate': 92, 'blood_pressure': '115/98', 'oxygen_saturation': 98}, 'setting': 'inpatient', 'patient_id': 365}
    )

    # Flu case (Patient ID: 1764)
    agent.add_document(
        """66-year-old male presents with cough, fatigue, and fever for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 38.3°C, heart rate 79 bpm, blood pressure 124/84, oxygen saturation 96%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 66, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'fatigue', 'fever', 'low_grade_fever'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 38.3, 'heart_rate': 79, 'blood_pressure': '124/84', 'oxygen_saturation': 96}, 'setting': 'inpatient', 'patient_id': 1764}
    )

    # Flu case (Patient ID: 1)
    agent.add_document(
        """74-year-old male presents with fatigue, sore throat, and fever for several days; reports feeling unwell with systemic symptoms, body temperature elevated to 39.4°C, heart rate 69 bpm, blood pressure 132/91, oxygen saturation 94%. Physical examination shows signs of viral illness with no focal respiratory findings. No complications noted. Provisional diagnosis: Flu. Differential: viral upper respiratory infection, seasonal influenza, early pneumonia (if chest findings develop), COVID-19 (pending testing). Plan: Supportive care with rest, adequate hydration, antipyretics for fever management, symptomatic treatment for comfort. Monitor for complications. Return if worsening dyspnea, persistent high fever >3 days, or signs of secondary bacterial infection. Severity: moderate presentation requiring medication and rest.""",
        source="clinical_database",
        metadata={'condition': 'Flu', 'age': 74, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'sore_throat', 'fever', 'high_fever', 'bradycardia'], 'severity': 'moderate', 'treatment': 'medication and rest', 'vitals': {'temperature': 39.4, 'heart_rate': 69, 'blood_pressure': '132/91', 'oxygen_saturation': 94}, 'setting': 'inpatient', 'patient_id': 1}
    )

    # Healthy case (Patient ID: 1209)
    agent.add_document(
        """19-year-old male presents with body ache, shortness of breath, and runny nose for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 37.9°C, heart rate 92 bpm, blood pressure 185/102, oxygen saturation 95% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 19, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'shortness_of_breath', 'runny_nose', 'low_grade_fever'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 37.9, 'heart_rate': 92, 'blood_pressure': '185/102', 'oxygen_saturation': 95}, 'setting': 'outpatient', 'patient_id': 1209}
    )

    # Healthy case (Patient ID: 1883)
    agent.add_document(
        """20-year-old male presents with runny nose, headache, and fatigue for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 36.7°C, heart rate 73 bpm, blood pressure 113/69, oxygen saturation 96% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 20, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['runny_nose', 'headache', 'fatigue'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.7, 'heart_rate': 73, 'blood_pressure': '113/69', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 1883}
    )

    # Healthy case (Patient ID: 1095)
    agent.add_document(
        """24-year-old female presents with sore throat, shortness of breath, and fatigue for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 39.3°C, heart rate 80 bpm, blood pressure 166/57, oxygen saturation 95% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 24, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['sore_throat', 'shortness_of_breath', 'fatigue', 'high_fever'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 39.3, 'heart_rate': 80, 'blood_pressure': '166/57', 'oxygen_saturation': 95}, 'setting': 'outpatient', 'patient_id': 1095}
    )

    # Healthy case (Patient ID: 1321)
    agent.add_document(
        """27-year-old female presents with body ache, fatigue, and shortness of breath for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 37.1°C, heart rate 65 bpm, blood pressure 149/77, oxygen saturation 96% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 27, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'fatigue', 'shortness_of_breath', 'bradycardia'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 37.1, 'heart_rate': 65, 'blood_pressure': '149/77', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 1321}
    )

    # Healthy case (Patient ID: 1423)
    agent.add_document(
        """44-year-old male presents with fatigue, body ache, and sore throat for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 37.2°C, heart rate 93 bpm, blood pressure 115/69, oxygen saturation 98% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 44, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'body_ache', 'sore_throat', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 37.2, 'heart_rate': 93, 'blood_pressure': '115/69', 'oxygen_saturation': 98}, 'setting': 'outpatient', 'patient_id': 1423}
    )

    # Healthy case (Patient ID: 1598)
    agent.add_document(
        """44-year-old female presents with shortness of breath, sore throat, and runny nose for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 38.8°C, heart rate 67 bpm, blood pressure 141/95, oxygen saturation 96% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 44, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['shortness_of_breath', 'sore_throat', 'runny_nose', 'low_grade_fever', 'bradycardia'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 38.8, 'heart_rate': 67, 'blood_pressure': '141/95', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 1598}
    )

    # Healthy case (Patient ID: 830)
    agent.add_document(
        """48-year-old male presents with headache, body ache, and shortness of breath for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 39.4°C, heart rate 90 bpm, blood pressure 104/95, oxygen saturation 96% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 48, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['headache', 'body_ache', 'shortness_of_breath', 'high_fever'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 39.4, 'heart_rate': 90, 'blood_pressure': '104/95', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 830}
    )

    # Healthy case (Patient ID: 1460)
    agent.add_document(
        """49-year-old female presents with body ache, runny nose, and shortness of breath for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 38.1°C, heart rate 65 bpm, blood pressure 139/62, oxygen saturation 97% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 49, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'runny_nose', 'shortness_of_breath', 'low_grade_fever', 'bradycardia', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 38.1, 'heart_rate': 65, 'blood_pressure': '139/62', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1460}
    )

    # Healthy case (Patient ID: 1633)
    agent.add_document(
        """61-year-old male presents with runny nose, shortness of breath, and sore throat for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 38.5°C, heart rate 76 bpm, blood pressure 134/103, oxygen saturation 97% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 61, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['runny_nose', 'shortness_of_breath', 'sore_throat', 'high_fever', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 38.5, 'heart_rate': 76, 'blood_pressure': '134/103', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1633}
    )

    # Healthy case (Patient ID: 2)
    agent.add_document(
        """66-year-old female presents with sore throat, fatigue, and cough for several days; presenting with mild constitutional symptoms, vital signs stable with temperature 39.0°C, heart rate 95 bpm, blood pressure 174/98, oxygen saturation 98% within normal limits. Overall appears well. Provisional diagnosis: Healthy. Differential: viral prodrome, mild dehydration, stress-related symptoms, early viral illness, non-specific malaise. Plan: Reassurance and supportive care, adequate rest and hydration, symptom monitoring. Return for evaluation if symptoms progress or new concerning features develop. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Healthy', 'age': 66, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['sore_throat', 'fatigue', 'cough', 'high_fever', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 39.0, 'heart_rate': 95, 'blood_pressure': '174/98', 'oxygen_saturation': 98}, 'setting': 'outpatient', 'patient_id': 2}
    )

    # Bronchitis case (Patient ID: 1201)
    agent.add_document(
        """20-year-old male presents with headache, cough, and runny nose for several days; persistent productive cough, temperature 37.2°C, heart rate 87 bpm, blood pressure 112/75, oxygen saturation 92% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 20, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['headache', 'cough', 'runny_nose', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 37.2, 'heart_rate': 87, 'blood_pressure': '112/75', 'oxygen_saturation': 92}, 'setting': 'inpatient', 'patient_id': 1201}
    )

    # Bronchitis case (Patient ID: 1808)
    agent.add_document(
        """20-year-old female presents with body ache, cough, and shortness of breath for several days; persistent productive cough, temperature 38.2°C, heart rate 94 bpm, blood pressure 106/103, oxygen saturation 93% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 20, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'cough', 'shortness_of_breath', 'low_grade_fever'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.2, 'heart_rate': 94, 'blood_pressure': '106/103', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 1808}
    )

    # Bronchitis case (Patient ID: 1120)
    agent.add_document(
        """23-year-old male presents with cough, sore throat, and headache for several days; persistent productive cough, temperature 38.5°C, heart rate 65 bpm, blood pressure 148/90, oxygen saturation 93% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 23, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'sore_throat', 'headache', 'high_fever', 'bradycardia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.5, 'heart_rate': 65, 'blood_pressure': '148/90', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 1120}
    )

    # Bronchitis case (Patient ID: 1844)
    agent.add_document(
        """25-year-old female presents with cough, runny nose, and sore throat for several days; persistent productive cough, temperature 37.6°C, heart rate 95 bpm, blood pressure 108/110, oxygen saturation 90% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 25, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'runny_nose', 'sore_throat', 'low_grade_fever', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 37.6, 'heart_rate': 95, 'blood_pressure': '108/110', 'oxygen_saturation': 90}, 'setting': 'inpatient', 'patient_id': 1844}
    )

    # Bronchitis case (Patient ID: 1157)
    agent.add_document(
        """35-year-old male presents with body ache, cough, and shortness of breath for several days; persistent productive cough, temperature 36.4°C, heart rate 91 bpm, blood pressure 110/68, oxygen saturation 94% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 35, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'cough', 'shortness_of_breath', 'low_temperature'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 36.4, 'heart_rate': 91, 'blood_pressure': '110/68', 'oxygen_saturation': 94}, 'setting': 'inpatient', 'patient_id': 1157}
    )

    # Bronchitis case (Patient ID: 1745)
    agent.add_document(
        """45-year-old female presents with cough, headache, and runny nose for several days; persistent productive cough, temperature 37.5°C, heart rate 96 bpm, blood pressure 147/88, oxygen saturation 93% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 45, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'headache', 'runny_nose'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 37.5, 'heart_rate': 96, 'blood_pressure': '147/88', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 1745}
    )

    # Bronchitis case (Patient ID: 1327)
    agent.add_document(
        """46-year-old male presents with cough, headache, and sore throat for several days; persistent productive cough, temperature 38.9°C, heart rate 88 bpm, blood pressure 106/90, oxygen saturation 92% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 46, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'headache', 'sore_throat', 'low_grade_fever', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.9, 'heart_rate': 88, 'blood_pressure': '106/90', 'oxygen_saturation': 92}, 'setting': 'inpatient', 'patient_id': 1327}
    )

    # Bronchitis case (Patient ID: 940)
    agent.add_document(
        """56-year-old female presents with body ache, cough, and runny nose for several days; persistent productive cough, temperature 37.1°C, heart rate 102 bpm, blood pressure 167/97, oxygen saturation 91% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 56, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'cough', 'runny_nose', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 37.1, 'heart_rate': 102, 'blood_pressure': '167/97', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 940}
    )

    # Bronchitis case (Patient ID: 1419)
    agent.add_document(
        """66-year-old male presents with body ache, cough, and shortness of breath for several days; persistent productive cough, temperature 39.5°C, heart rate 64 bpm, blood pressure 178/75, oxygen saturation 91% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 66, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'cough', 'shortness_of_breath', 'high_fever', 'bradycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.5, 'heart_rate': 64, 'blood_pressure': '178/75', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1419}
    )

    # Bronchitis case (Patient ID: 7)
    agent.add_document(
        """21-year-old male presents with sore throat, fatigue, and cough for several days; persistent productive cough, temperature 37.5°C, heart rate 71 bpm, blood pressure 126/82, oxygen saturation 93% below normal. Clinical presentation suggests lower respiratory tract involvement. Provisional diagnosis: Bronchitis. Differential: acute bronchitis, pneumonia (if fever and consolidation), asthma exacerbation, COPD exacerbation (if history), whooping cough. Plan: Bronchodilators if wheezing present, supportive care, adequate hydration, cough suppressants if dry cough interfering with sleep. Antibiotics only if bacterial superinfection suspected. Hospitalization and medication required given severity. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Bronchitis', 'age': 21, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['sore_throat', 'fatigue', 'cough'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 37.5, 'heart_rate': 71, 'blood_pressure': '126/82', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 7}
    )

    # Cold case (Patient ID: 1067)
    agent.add_document(
        """26-year-old male presents with fever, headache, and runny nose for several days; mild upper respiratory symptoms, low-grade temperature 36.2°C, stable vital signs with heart rate 61 bpm, blood pressure 152/71, good oxygen saturation 96%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 26, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fever', 'headache', 'runny_nose', 'low_temperature', 'bradycardia'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.2, 'heart_rate': 61, 'blood_pressure': '152/71', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 1067}
    )

    # Cold case (Patient ID: 1932)
    agent.add_document(
        """25-year-old female presents with headache, body ache, and fatigue for several days; mild upper respiratory symptoms, low-grade temperature 36.4°C, stable vital signs with heart rate 119 bpm, blood pressure 103/104, good oxygen saturation 98%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 25, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['headache', 'body_ache', 'fatigue', 'low_temperature', 'tachycardia', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.4, 'heart_rate': 119, 'blood_pressure': '103/104', 'oxygen_saturation': 98}, 'setting': 'outpatient', 'patient_id': 1932}
    )

    # Cold case (Patient ID: 1575)
    agent.add_document(
        """29-year-old female presents with body ache, fatigue, and sore throat for several days; mild upper respiratory symptoms, low-grade temperature 35.8°C, stable vital signs with heart rate 96 bpm, blood pressure 163/95, good oxygen saturation 99%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 29, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'fatigue', 'sore_throat', 'low_temperature', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 35.8, 'heart_rate': 96, 'blood_pressure': '163/95', 'oxygen_saturation': 99}, 'setting': 'outpatient', 'patient_id': 1575}
    )

    # Cold case (Patient ID: 1537)
    agent.add_document(
        """34-year-old male presents with cough, headache, and shortness of breath for several days; mild upper respiratory symptoms, low-grade temperature 36.4°C, stable vital signs with heart rate 83 bpm, blood pressure 155/78, good oxygen saturation 98%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 34, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'headache', 'shortness_of_breath', 'low_temperature', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.4, 'heart_rate': 83, 'blood_pressure': '155/78', 'oxygen_saturation': 98}, 'setting': 'outpatient', 'patient_id': 1537}
    )

    # Cold case (Patient ID: 1295)
    agent.add_document(
        """39-year-old female presents with body ache, cough, and shortness of breath for several days; mild upper respiratory symptoms, low-grade temperature 36.4°C, stable vital signs with heart rate 117 bpm, blood pressure 91/112, good oxygen saturation 97%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 39, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'cough', 'shortness_of_breath', 'low_temperature', 'tachycardia', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.4, 'heart_rate': 117, 'blood_pressure': '91/112', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1295}
    )

    # Cold case (Patient ID: 1386)
    agent.add_document(
        """45-year-old male presents with cough, fatigue, and sore throat for several days; mild upper respiratory symptoms, low-grade temperature 36.1°C, stable vital signs with heart rate 75 bpm, blood pressure 126/80, good oxygen saturation 97%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 45, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'fatigue', 'sore_throat', 'low_temperature', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.1, 'heart_rate': 75, 'blood_pressure': '126/80', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1386}
    )

    # Cold case (Patient ID: 1851)
    agent.add_document(
        """50-year-old female presents with cough, headache, and shortness of breath for several days; mild upper respiratory symptoms, low-grade temperature 35.8°C, stable vital signs with heart rate 95 bpm, blood pressure 141/94, good oxygen saturation 97%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 50, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'headache', 'shortness_of_breath', 'low_temperature', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 35.8, 'heart_rate': 95, 'blood_pressure': '141/94', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1851}
    )

    # Cold case (Patient ID: 1498)
    agent.add_document(
        """55-year-old male presents with body ache, shortness of breath, and fever for several days; mild upper respiratory symptoms, low-grade temperature 36.1°C, stable vital signs with heart rate 73 bpm, blood pressure 154/87, good oxygen saturation 96%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 55, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['body_ache', 'shortness_of_breath', 'fever', 'low_temperature'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.1, 'heart_rate': 73, 'blood_pressure': '154/87', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 1498}
    )

    # Cold case (Patient ID: 1715)
    agent.add_document(
        """68-year-old female presents with headache, runny nose, and fever for several days; mild upper respiratory symptoms, low-grade temperature 35.9°C, stable vital signs with heart rate 112 bpm, blood pressure 148/115, good oxygen saturation 97%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 68, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['headache', 'runny_nose', 'fever', 'low_temperature', 'tachycardia', 'normal_oxygenation'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 35.9, 'heart_rate': 112, 'blood_pressure': '148/115', 'oxygen_saturation': 97}, 'setting': 'outpatient', 'patient_id': 1715}
    )

    # Cold case (Patient ID: 9)
    agent.add_document(
        """56-year-old female presents with shortness of breath, fever, and headache for several days; mild upper respiratory symptoms, low-grade temperature 36.2°C, stable vital signs with heart rate 103 bpm, blood pressure 152/71, good oxygen saturation 96%. Self-limited viral illness pattern. Provisional diagnosis: Cold. Differential: common cold (rhinovirus), allergic rhinitis, early influenza, sinusitis (if facial pain/pressure develops). Plan: Conservative management with rest, increased fluid intake, humidified air, saline nasal rinses. Symptomatic relief with decongestants if needed. Expected resolution within 7-10 days. Severity: mild presentation requiring rest and fluids.""",
        source="clinical_database",
        metadata={'condition': 'Cold', 'age': 56, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['shortness_of_breath', 'fever', 'headache', 'low_temperature', 'tachycardia'], 'severity': 'mild', 'treatment': 'rest and fluids', 'vitals': {'temperature': 36.2, 'heart_rate': 103, 'blood_pressure': '152/71', 'oxygen_saturation': 96}, 'setting': 'outpatient', 'patient_id': 9}
    )

    # Pneumonia case (Patient ID: 1218)
    agent.add_document(
        """22-year-old male presents with fatigue, fever, and runny nose for several days; acute onset with high fever 39.8°C, tachycardia 101 bpm, blood pressure 100/67, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 22, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'runny_nose', 'high_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.8, 'heart_rate': 101, 'blood_pressure': '100/67', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1218}
    )

    # Pneumonia case (Patient ID: 1773)
    agent.add_document(
        """23-year-old male presents with fatigue, fever, and headache for several days; acute onset with high fever 39.7°C, tachycardia 106 bpm, blood pressure 112/73, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 23, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'headache', 'high_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.7, 'heart_rate': 106, 'blood_pressure': '112/73', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1773}
    )

    # Pneumonia case (Patient ID: 1994)
    agent.add_document(
        """24-year-old female presents with fever, headache, and shortness of breath for several days; acute onset with high fever 38.7°C, tachycardia 119 bpm, blood pressure 114/92, reduced oxygen saturation 92% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 24, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fever', 'headache', 'shortness_of_breath', 'low_grade_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.7, 'heart_rate': 119, 'blood_pressure': '114/92', 'oxygen_saturation': 92}, 'setting': 'inpatient', 'patient_id': 1994}
    )

    # Pneumonia case (Patient ID: 1993)
    agent.add_document(
        """25-year-old female presents with fatigue, fever, and shortness of breath for several days; acute onset with high fever 39.7°C, tachycardia 117 bpm, blood pressure 106/81, reduced oxygen saturation 94% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 25, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'shortness_of_breath', 'high_fever', 'tachycardia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.7, 'heart_rate': 117, 'blood_pressure': '106/81', 'oxygen_saturation': 94}, 'setting': 'inpatient', 'patient_id': 1993}
    )

    # Pneumonia case (Patient ID: 1779)
    agent.add_document(
        """41-year-old male presents with cough, fever, and shortness of breath for several days; acute onset with high fever 39.9°C, tachycardia 106 bpm, blood pressure 124/77, reduced oxygen saturation 90% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 41, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['cough', 'fever', 'shortness_of_breath', 'high_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.9, 'heart_rate': 106, 'blood_pressure': '124/77', 'oxygen_saturation': 90}, 'setting': 'inpatient', 'patient_id': 1779}
    )

    # Pneumonia case (Patient ID: 1925)
    agent.add_document(
        """43-year-old female presents with fatigue, fever, and headache for several days; acute onset with high fever 39.0°C, tachycardia 114 bpm, blood pressure 120/93, reduced oxygen saturation 93% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 43, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'headache', 'high_fever', 'tachycardia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.0, 'heart_rate': 114, 'blood_pressure': '120/93', 'oxygen_saturation': 93}, 'setting': 'inpatient', 'patient_id': 1925}
    )

    # Pneumonia case (Patient ID: 1974)
    agent.add_document(
        """46-year-old male presents with fatigue, fever, and runny nose for several days; acute onset with high fever 38.6°C, tachycardia 105 bpm, blood pressure 122/64, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 46, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'runny_nose', 'low_grade_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.6, 'heart_rate': 105, 'blood_pressure': '122/64', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1974}
    )

    # Pneumonia case (Patient ID: 1665)
    agent.add_document(
        """67-year-old female presents with fatigue, fever, and headache for several days; acute onset with high fever 39.1°C, tachycardia 115 bpm, blood pressure 165/97, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 67, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'headache', 'high_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.1, 'heart_rate': 115, 'blood_pressure': '165/97', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 1665}
    )

    # Pneumonia case (Patient ID: 1340)
    agent.add_document(
        """68-year-old male presents with fever, headache, and shortness of breath for several days; acute onset with high fever 38.9°C, tachycardia 116 bpm, blood pressure 107/98, reduced oxygen saturation 90% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 68, 'sex': 'male', 'body_system': 'respiratory', 'symptom_tags': ['fever', 'headache', 'shortness_of_breath', 'low_grade_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 38.9, 'heart_rate': 116, 'blood_pressure': '107/98', 'oxygen_saturation': 90}, 'setting': 'inpatient', 'patient_id': 1340}
    )

    # Pneumonia case (Patient ID: 436)
    agent.add_document(
        """19-year-old female presents with fatigue, fever, and runny nose for several days; acute onset with high fever 39.8°C, tachycardia 101 bpm, blood pressure 100/67, reduced oxygen saturation 91% concerning for pneumonia. Physical exam may reveal crackles, bronchial breathing, or dullness to percussion. Provisional diagnosis: Pneumonia. Differential: bacterial pneumonia, viral pneumonia, atypical pneumonia, pulmonary embolism (if sudden onset), pneumothorax. Plan: Immediate antibiotic therapy, supplemental oxygen if SpO2 <92%, chest X-ray for confirmation, blood cultures, consider hospitalization given severity. Close monitoring of vital signs and oxygenation. Severity: severe presentation requiring hospitalization and medication.""",
        source="clinical_database",
        metadata={'condition': 'Pneumonia', 'age': 19, 'sex': 'female', 'body_system': 'respiratory', 'symptom_tags': ['fatigue', 'fever', 'runny_nose', 'high_fever', 'tachycardia', 'hypoxemia'], 'severity': 'severe', 'treatment': 'hospitalization and medication', 'vitals': {'temperature': 39.8, 'heart_rate': 101, 'blood_pressure': '100/67', 'oxygen_saturation': 91}, 'setting': 'inpatient', 'patient_id': 436}
    )
# append_inline_docs_into_existing.py
# Appends 50 inline documents into existing JSON + FAISS in-place
    # Optional: stats after append
    try:
        stats2 = agent.get_stats()
        print(f"After append: {stats2.get('total_documents')} docs, {stats2.get('total_chunks')} chunks")
    except:
        print("Appended 50 inline documents.")

    # 3) Save back to the same files
    agent.save_memory(EXISTING_JSON)
    # If your implementation separates FAISS saving, also:
    # agent.save_faiss(EXISTING_FAISS)

    print(f"Saved updated memory to {EXISTING_JSON} and FAISS to {EXISTING_FAISS}")

if __name__ == "__main__":
    main()
