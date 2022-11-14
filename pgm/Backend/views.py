from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import BayesianNetwork
from IPython.display import Image, display
from pgmpy.estimators import HillClimbSearch, BicScore, PC, K2Score
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Create your views here.

def index(request):
    # Launch the Landing Page
    return render(request, "index.html")

def finish(request):
    # Recieves the parameters and Process and Send Output.

    print (request.POST)
    
    anxiety = Anxiety(request.POST)
    depression = 0

    if(anxiety > 0.7):
        depression = Depression(request.POST, 1)
    else:
        depression = Depression(request.POST, 0)

    print(anxiety)
    print(depression)


    return render(request, 'finish2.html', {'Anxiety': anxiety, 'Depression': depression})

    ## For JSON Output
    # return JsonResponse({'Response' : request.POST.get('your_name')})

    
def Depression(dic, ax_val):
    df = pd.read_csv('Database.csv',encoding = 'utf-8')
    lbl=LabelEncoder()
    for col in df.columns:
        df[col]=lbl.fit_transform(df[[col]])
    model = BayesianNetwork()
    model.add_edges_from([('Age (4 levels)', 'Public health insurance '),
 ('Age (4 levels)', 'Field of study'),
 ('Age (4 levels)', 'French nationality'),
 ('Gender', 'Weight (kg)'),
 ('Gender', 'Field of study'),
 ('Gender', 'Urinalysis leukocyturia)'),
 ('Gender', 'Binge drinking'),
 ('Gender', 'On a diet'),
 ('French nationality', 'Vaccination up to date'),
 ('French nationality', 'Private health insurance '),
 ('French nationality', 'Grant'),
 ('Field of study', 'Physical activity(3 levels)'),
 ('Field of study', 'Public health insurance '),
 ('Field of study', 'Professional objective'),
 ('Field of study', 'Vaccination up to date'),
 ('Field of study', 'Additional income'),
 ('Year of university', 'Age (4 levels)'),
 ('Year of university', 'Field of study'),
 ('Learning disabilities', 'Depressive symptoms'),
 ('Learning disabilities', 'Unbalanced meals'),
 ('Learning disabilities', 'Vaccination up to date'),
 ('Learning disabilities', 'Informed about opportunities'),
 ('Learning disabilities', 'Eating junk food'),
 ('Professional objective', 'Informed about opportunities'),
 ('Professional objective', 'Irregular rhythm of meals'),
 ('Professional objective', 'Long commute'),
 ('Informed about opportunities', 'Physical activity(3 levels)'),
 ('Informed about opportunities', 'Irregular rhythm of meals'),
 ('Satisfied with living conditions', 'Informed about opportunities'),
 ('Satisfied with living conditions', 'Parental home'),
 ('Satisfied with living conditions', 'Professional objective'),
 ('Satisfied with living conditions', 'Long commute'),
 ('Satisfied with living conditions', 'Year of university'),
 ('Living with a partner/child', 'Long commute'),
 ('Living with a partner/child', 'Siblings'),
 ('Living with a partner/child', 'Eating junk food'),
 ('Parental home', 'Long commute'),
 ('Parental home', 'Cigarette smoker (5 levels)'),
 ('Parental home', 'Irregular rhythm of meals'),
 ('Parental home', 'Field of study'),
 ('Parental home', 'Marijuana use'),
 ('Parental home', 'Unbalanced meals'),
 ('Parental home', 'Mode of transportation'),
 ('Parental home', 'Age (4 levels)'),
 ('Parental home', 'Vaccination up to date'),
 ('Parental home', 'Grant'),
 ('Parental home', 'Year of university'),
 ('Having only one parent', 'Siblings'),
 ('Having only one parent', 'Long commute'),
 ('Having only one parent', 'Grant'),
 ('Having only one parent', 'At least one parent unemployed'),
 ('Having only one parent', 'Private health insurance '),
 ('Long commute', 'Mode of transportation'),
 ('Long commute', 'Eating junk food'),
 ('Long commute', 'Urinalysis leukocyturia)'),
 ('Mode of transportation', 'Eating junk food'),
 ('Financial difficulties', 'Learning disabilities'),
 ('Financial difficulties', 'Irregular rhythm of meals'),
 ('Financial difficulties', 'French nationality'),
 ('Financial difficulties', 'Depressive symptoms'),
 ('Financial difficulties', 'Field of study'),
 ('Financial difficulties', 'Mode of transportation'),
 ('Financial difficulties', 'Physical activity(3 levels)'),
 ('Financial difficulties', 'Vaccination up to date'),
 ('Grant', 'Public health insurance '),
 ('Grant', 'Siblings'),
 ('Grant', 'At least one parent unemployed'),
 ('Additional income', 'Irregular rhythm of meals'),
 ('Private health insurance ', 'C.M.U.'),
 ('Private health insurance ', 'Living with a partner/child'),
 ('C.M.U.', 'At least one parent unemployed'),
 ('C.M.U.', 'Grant'),
 ('Irregular rhythm of meals', 'Irregular rhythm or unbalanced meals'),
 ('Irregular rhythm of meals', 'Unbalanced meals'),
 ('Unbalanced meals', 'Physical activity(3 levels)'),
 ('Unbalanced meals', 'Irregular rhythm or unbalanced meals'),
 ('Eating junk food', 'Irregular rhythm of meals'),
 ('Eating junk food', 'Unbalanced meals'),
 ('On a diet', 'Control examination needed'),
 ('On a diet', 'Eating junk food'),
 ('Height (cm)', 'Gender'),
 ('Overweight and obesity', 'Weight (kg)'),
 ('Overweight and obesity', 'Height (cm)'),
 ('Systolic blood pressure (mmHg)', 'Abnormal heart rate'),
 ('Diastolic blood pressure (mmHg)', 'Abnormal heart rate'),
 ('Diastolic blood pressure (mmHg)', 'Prehypertension or hypertension'),
 ('Diastolic blood pressure (mmHg)', 'Systolic blood pressure (mmHg)'),
 ('Prehypertension or hypertension', 'Systolic blood pressure (mmHg)'),
 ('Abnormal heart rate', 'Heart rate (bpm)'),
 ('Abnormal heart rate', 'Overweight and obesity'),
 ('Abnormal heart rate', 'Satisfied with living conditions'),
 ('Distant visual acuity of left eye (score /10)',
  'Decreased in distant visual acuity'),
 ('Distant visual acuity of left eye (score /10)',
  'Control examination needed'),
 ('Distant visual acuity of left eye (score /10)',
  'Distant visual acuity of right eye (score /10)'),
 ('Close visual acuity of right eye (score /10)',
  'Close visual acuity of left eye (score /10)'),
 ('Close visual acuity of left eye (score /10)',
  'Decreased in close visual acuity'),
 ('Decreased in distant visual acuity',
  'Distant visual acuity of right eye (score /10)'),
 ('Decreased in close visual acuity',
  'Distant visual acuity of left eye (score /10)'),
 ('Decreased in close visual acuity', 'Urinalysis leukocyturia)'),
 ('Urinalysis (hematuria)', 'Eating junk food'),
 ('Urinalysis leukocyturia)', 'Abnormal urinalysis'),
 ('Urinalysis leukocyturia)', 'Urinalysis (hematuria)'),
 ('Urinalysis leukocyturia)', 'Eating junk food'),
 ('Abnormal urinalysis', 'Urinalysis (hematuria)'),
 ('Vaccination up to date', 'Control examination needed'),
 ('Control examination needed', 'Other recreational drugs'),
 ('Anxiety symptoms', 'Depressive symptoms'),
 ('Anxiety symptoms', 'Financial difficulties'),
 ('Anxiety symptoms', 'Difficulty memorizing lessons'),
 ('Anxiety symptoms', 'Control examination needed'),
 ('Panic attack symptoms', 'Anxiety symptoms'),
 ('Panic attack symptoms', 'Depressive symptoms'),
 ('Panic attack symptoms', 'Irregular rhythm of meals'),
 ('Panic attack symptoms', 'Eating junk food'),
 ('Depressive symptoms', 'Difficulty memorizing lessons'),
 ('Depressive symptoms', 'Satisfied with living conditions'),
 ('Depressive symptoms', 'Parental home'),
 ('Cigarette smoker (5 levels)', 'Marijuana use'),
 ('Cigarette smoker (5 levels)', 'Drinker (3 levels)'),
 ('Cigarette smoker (5 levels)', 'Overweight and obesity'),
 ('Cigarette smoker (5 levels)', 'Control examination needed'),
 ('Drinker (3 levels)', 'Binge drinking'),
 ('Drinker (3 levels)', 'Grant'),
 ('Drinker (3 levels)', 'Marijuana use'),
 ('Drinker (3 levels)', 'French nationality'),
 ('Binge drinking', 'Other recreational drugs'),
 ('Marijuana use', 'Other recreational drugs'),
 ('Marijuana use', 'Eating junk food'),
 ('Marijuana use', 'Having only one parent'),
 ('Other recreational drugs', 'Informed about opportunities')]) 
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
    inference = VariableElimination(model)
    hj=dict(df['Weight (kg)'])
    index = list(hj.keys())
    for i in range(len(df['Weight (kg)'])):
        if(df['Weight (kg)'][index[i]]>=38 and df['Weight (kg)'][index[i]]<=52):
            df['Weight (kg)'][index[i]]=1
        elif(df['Weight (kg)'][index[i]]>=53 and df['Weight (kg)'][index[i]]<=67):
            df['Weight (kg)'][index[i]]=2
        elif(df['Weight (kg)'][index[i]]>=68 and df['Weight (kg)'][index[i]]<=82):
            df['Weight (kg)'][index[i]]=3
        elif(df['Weight (kg)'][index[i]]>=83 and df['Weight (kg)'][index[i]]<=97):
            df['Weight (kg)'][index[i]]=4
        elif(df['Weight (kg)'][index[i]]>=98 and df['Weight (kg)'][index[i]]<=112):
            df['Weight (kg)'][index[i]]=5
        elif(df['Weight (kg)'][index[i]]>=113 and df['Weight (kg)'][index[i]]<=127):
            df['Weight (kg)'][index[i]]=6
        elif(df['Weight (kg)'][index[i]]>=128 and df['Weight (kg)'][index[i]]<=142):
            df['Weight (kg)'][index[i]]=7

    hj=dict(df['Systolic blood pressure (mmHg)'])
    index = list(hj.keys())
    for i in range(len(df['Systolic blood pressure (mmHg)'])):
        if(df['Systolic blood pressure (mmHg)'][index[i]]>=8 and df['Systolic blood pressure (mmHg)'][index[i]]<=11):
            df['Systolic blood pressure (mmHg)'][index[i]]=1
        elif(df['Systolic blood pressure (mmHg)'][index[i]]>11 and df['Systolic blood pressure (mmHg)'][index[i]]<=14):
            df['Systolic blood pressure (mmHg)'][index[i]]=2
        elif(df['Systolic blood pressure (mmHg)'][index[i]]>14 and df['Systolic blood pressure (mmHg)'][index[i]]<=17):
            df['Systolic blood pressure (mmHg)'][index[i]]=3 

    hj=dict(df['Diastolic blood pressure (mmHg)'])
    index = list(hj.keys())
    for i in range(len(df['Diastolic blood pressure (mmHg)'])):
        if(df['Diastolic blood pressure (mmHg)'][index[i]]>=4 and df['Diastolic blood pressure (mmHg)'][index[i]]<7):
            df['Diastolic blood pressure (mmHg)'][index[i]]=1
        elif(df['Diastolic blood pressure (mmHg)'][index[i]]>=7 and df['Diastolic blood pressure (mmHg)'][index[i]]<10):
            df['Diastolic blood pressure (mmHg)'][index[i]]=2
        elif(df['Diastolic blood pressure (mmHg)'][index[i]]>=10 and df['Diastolic blood pressure (mmHg)'][index[i]]<=13):
            df['Diastolic blood pressure (mmHg)'][index[i]]=3

    hj=dict(df['Heart rate (bpm)'])
    index = list(hj.keys())
    for i in range(len(df['Heart rate (bpm)'])):
        if(df['Heart rate (bpm)'][index[i]]>=40 and df['Heart rate (bpm)'][index[i]]<=54):
            df['Heart rate (bpm)'][index[i]]=1
        elif(df['Heart rate (bpm)'][index[i]]>=55 and df['Heart rate (bpm)'][index[i]]<=69):
            df['Heart rate (bpm)'][index[i]]=2
        elif(df['Heart rate (bpm)'][index[i]]>=70 and df['Heart rate (bpm)'][index[i]]<=84):
            df['Heart rate (bpm)'][index[i]]=3
        elif(df['Heart rate (bpm)'][index[i]]>=85 and df['Heart rate (bpm)'][index[i]]<=99):
            df['Heart rate (bpm)'][index[i]]=4
        elif(df['Heart rate (bpm)'][index[i]]>=100 and df['Heart rate (bpm)'][index[i]]<=114):
            df['Heart rate (bpm)'][index[i]]=5
        elif(df['Heart rate (bpm)'][index[i]]>=115 and df['Heart rate (bpm)'][index[i]]<=129):
            df['Heart rate (bpm)'][index[i]]=6

    q = inference.query(['Depressive symptoms'],evidence={
                                                      'Age (4 levels)': int(dic['age']),
                                                      'Gender': int(dic['Gender']) ,
                                                      'French nationality': int(dic['french_nationality']) ,             
                                                      'Field of study': int(dic['study']) ,
                                                      'Year of university': int(dic['year']) ,
                                                      'Learning disabilities': int(dic['l_disability']) ,
                                                      'Difficulty memorizing lessons': int(dic['d_mem_lessons']) ,
                                                      'Professional objective': int(dic['p_objective']) ,
                                                      'Informed about opportunities': int(dic['opportunities']) ,
                                                      'Satisfied with living conditions':int(dic['l_conditions']) ,
                                                      'Living with a partner/child': int(dic['l_partner']) ,
                                                      'Parental home': int(dic['p_home']) ,
                                                      'Having only one parent': int(dic['one_parent']) ,
                                                      'At least one parent unemployed': int(dic['p_unemployed']) ,
                                                      'Siblings': int(dic['siblings']) ,
                                                      'Long commute': int(dic['l_commute']) ,
                                                      'Mode of transportation': int(dic['m_transport']) ,
                                                      'Financial difficulties': int(dic['f_difficulties']) ,
                                                      'Grant': int(dic['grant']) ,
                                                      'Additional income': int(dic['a_income'])  ,
                                                      'Public health insurance ': int(dic['public_insurance']) ,
                                                      'Private health insurance ': int(dic['private_insurance']) ,
                                                      'C.M.U.': int(dic['cmu']) ,
                                                      'Irregular rhythm of meals': int(dic['i_meals']) ,
                                                      'Unbalanced meals': int(dic['u_meals']) ,
                                                      'Eating junk food': int(dic['junk_food']) ,
                                                      'On a diet':  int(dic['diet']) ,
                                                      'Irregular rhythm or unbalanced meals': int(dic['i_u_meals']) ,
                                                      'Physical activity(3 levels)': int(dic['p_activity']) ,
                                                      'Weight (kg)': int(dic['weight']) ,
                                                      'Height (cm)': int(dic['height']) ,
                                                      'Overweight and obesity':  int(dic['overweight']) ,
                                                      'Systolic blood pressure (mmHg)': int(dic['s_bp']) ,
                                                      'Diastolic blood pressure (mmHg)': int(dic['d_bp']) ,
                                                      'Prehypertension or hypertension': int(dic['hypertension']) ,
                                                      'Heart rate (bpm)': int(dic['heartrate']) ,
                                                      'Abnormal heart rate': int(dic['a_heartrate']) ,
                                                      'Distant visual acuity of right eye (score /10)': int(dic['act_1']) ,
                                                      'Distant visual acuity of left eye (score /10)': int(dic['act_2']) ,
                                                      'Close visual acuity of right eye (score /10)': int(dic['act_3']) ,
                                                      'Close visual acuity of left eye (score /10)': int(dic['act_4']) ,
                                                      'Decreased in distant visual acuity': int(dic['act_5']) ,
                                                      'Decreased in close visual acuity': int(dic['act_6']) ,
                                                      'Urinalysis (hematuria)': int(dic['u_h']) ,
                                                      'Urinalysis leukocyturia)': int(dic['u_l']) ,
                                                      'Abnormal urinalysis': int(dic['abnormal_u']) ,
                                                      'Vaccination up to date': int(dic['vaccine']) ,
                                                      'Control examination needed': int(dic['control_exam']) ,
                                                      'Panic attack symptoms': int(dic['panic_atck']) ,
                                                      'Cigarette smoker (5 levels)': int(dic['c_smoker']) ,
                                                      'Drinker (3 levels)': int(dic['drinker']) ,
                                                      'Binge drinking': int(dic['b_drinking']) ,
                                                      'Marijuana use': int(dic['marjiuana']) ,
                                                      'Other recreational drugs':  int(dic['o_drugs']) ,
                                                      'Anxiety symptoms':ax_val,      
                                            }, joint=False)['Depressive symptoms']
    return q.values[1]

def Anxiety(dic):
    df = pd.read_csv('Database.csv',encoding = 'utf-8')
    lbl=LabelEncoder()
    for col in df.columns:
        df[col]=lbl.fit_transform(df[[col]])
    model = BayesianNetwork()
    model.add_edges_from([('Age (4 levels)', 'Public health insurance '),
 ('Age (4 levels)', 'Field of study'),
 ('Age (4 levels)', 'French nationality'),
 ('Gender', 'Weight (kg)'),
 ('Gender', 'Field of study'),
 ('Gender', 'Urinalysis leukocyturia)'),
 ('Gender', 'Binge drinking'),
 ('Gender', 'On a diet'),
 ('French nationality', 'Vaccination up to date'),
 ('French nationality', 'Private health insurance '),
 ('French nationality', 'Grant'),
 ('Field of study', 'Physical activity(3 levels)'),
 ('Field of study', 'Public health insurance '),
 ('Field of study', 'Professional objective'),
 ('Field of study', 'Vaccination up to date'),
 ('Field of study', 'Additional income'),
 ('Year of university', 'Age (4 levels)'),
 ('Year of university', 'Field of study'),
 ('Learning disabilities', 'Depressive symptoms'),
 ('Learning disabilities', 'Unbalanced meals'),
 ('Learning disabilities', 'Vaccination up to date'),
 ('Learning disabilities', 'Informed about opportunities'),
 ('Learning disabilities', 'Eating junk food'),
 ('Professional objective', 'Informed about opportunities'),
 ('Professional objective', 'Irregular rhythm of meals'),
 ('Professional objective', 'Long commute'),
 ('Informed about opportunities', 'Physical activity(3 levels)'),
 ('Informed about opportunities', 'Irregular rhythm of meals'),
 ('Satisfied with living conditions', 'Informed about opportunities'),
 ('Satisfied with living conditions', 'Parental home'),
 ('Satisfied with living conditions', 'Professional objective'),
 ('Satisfied with living conditions', 'Long commute'),
 ('Satisfied with living conditions', 'Year of university'),
 ('Living with a partner/child', 'Long commute'),
 ('Living with a partner/child', 'Siblings'),
 ('Living with a partner/child', 'Eating junk food'),
 ('Parental home', 'Long commute'),
 ('Parental home', 'Cigarette smoker (5 levels)'),
 ('Parental home', 'Irregular rhythm of meals'),
 ('Parental home', 'Field of study'),
 ('Parental home', 'Marijuana use'),
 ('Parental home', 'Unbalanced meals'),
 ('Parental home', 'Mode of transportation'),
 ('Parental home', 'Age (4 levels)'),
 ('Parental home', 'Vaccination up to date'),
 ('Parental home', 'Grant'),
 ('Parental home', 'Year of university'),
 ('Having only one parent', 'Siblings'),
 ('Having only one parent', 'Long commute'),
 ('Having only one parent', 'Grant'),
 ('Having only one parent', 'At least one parent unemployed'),
 ('Having only one parent', 'Private health insurance '),
 ('Long commute', 'Mode of transportation'),
 ('Long commute', 'Eating junk food'),
 ('Long commute', 'Urinalysis leukocyturia)'),
 ('Mode of transportation', 'Eating junk food'),
 ('Financial difficulties', 'Learning disabilities'),
 ('Financial difficulties', 'Irregular rhythm of meals'),
 ('Financial difficulties', 'French nationality'),
 ('Financial difficulties', 'Depressive symptoms'),
 ('Financial difficulties', 'Field of study'),
 ('Financial difficulties', 'Mode of transportation'),
 ('Financial difficulties', 'Physical activity(3 levels)'),
 ('Financial difficulties', 'Vaccination up to date'),
 ('Grant', 'Public health insurance '),
 ('Grant', 'Siblings'),
 ('Grant', 'At least one parent unemployed'),
 ('Additional income', 'Irregular rhythm of meals'),
 ('Private health insurance ', 'C.M.U.'),
 ('Private health insurance ', 'Living with a partner/child'),
 ('C.M.U.', 'At least one parent unemployed'),
 ('C.M.U.', 'Grant'),
 ('Irregular rhythm of meals', 'Irregular rhythm or unbalanced meals'),
 ('Irregular rhythm of meals', 'Unbalanced meals'),
 ('Unbalanced meals', 'Physical activity(3 levels)'),
 ('Unbalanced meals', 'Irregular rhythm or unbalanced meals'),
 ('Eating junk food', 'Irregular rhythm of meals'),
 ('Eating junk food', 'Unbalanced meals'),
 ('On a diet', 'Control examination needed'),
 ('On a diet', 'Eating junk food'),
 ('Height (cm)', 'Gender'),
 ('Overweight and obesity', 'Weight (kg)'),
 ('Overweight and obesity', 'Height (cm)'),
 ('Systolic blood pressure (mmHg)', 'Abnormal heart rate'),
 ('Diastolic blood pressure (mmHg)', 'Abnormal heart rate'),
 ('Diastolic blood pressure (mmHg)', 'Prehypertension or hypertension'),
 ('Diastolic blood pressure (mmHg)', 'Systolic blood pressure (mmHg)'),
 ('Prehypertension or hypertension', 'Systolic blood pressure (mmHg)'),
 ('Abnormal heart rate', 'Heart rate (bpm)'),
 ('Abnormal heart rate', 'Overweight and obesity'),
 ('Abnormal heart rate', 'Satisfied with living conditions'),
 ('Distant visual acuity of left eye (score /10)',
  'Decreased in distant visual acuity'),
 ('Distant visual acuity of left eye (score /10)',
  'Control examination needed'),
 ('Distant visual acuity of left eye (score /10)',
  'Distant visual acuity of right eye (score /10)'),
 ('Close visual acuity of right eye (score /10)',
  'Close visual acuity of left eye (score /10)'),
 ('Close visual acuity of left eye (score /10)',
  'Decreased in close visual acuity'),
 ('Decreased in distant visual acuity',
  'Distant visual acuity of right eye (score /10)'),
 ('Decreased in close visual acuity',
  'Distant visual acuity of left eye (score /10)'),
 ('Decreased in close visual acuity', 'Urinalysis leukocyturia)'),
 ('Urinalysis (hematuria)', 'Eating junk food'),
 ('Urinalysis leukocyturia)', 'Abnormal urinalysis'),
 ('Urinalysis leukocyturia)', 'Urinalysis (hematuria)'),
 ('Urinalysis leukocyturia)', 'Eating junk food'),
 ('Abnormal urinalysis', 'Urinalysis (hematuria)'),
 ('Vaccination up to date', 'Control examination needed'),
 ('Control examination needed', 'Other recreational drugs'),
 ('Anxiety symptoms', 'Depressive symptoms'),
 ('Anxiety symptoms', 'Financial difficulties'),
 ('Anxiety symptoms', 'Difficulty memorizing lessons'),
 ('Anxiety symptoms', 'Control examination needed'),
 ('Panic attack symptoms', 'Anxiety symptoms'),
 ('Panic attack symptoms', 'Depressive symptoms'),
 ('Panic attack symptoms', 'Irregular rhythm of meals'),
 ('Panic attack symptoms', 'Eating junk food'),
 ('Depressive symptoms', 'Difficulty memorizing lessons'),
 ('Depressive symptoms', 'Satisfied with living conditions'),
 ('Depressive symptoms', 'Parental home'),
 ('Cigarette smoker (5 levels)', 'Marijuana use'),
 ('Cigarette smoker (5 levels)', 'Drinker (3 levels)'),
 ('Cigarette smoker (5 levels)', 'Overweight and obesity'),
 ('Cigarette smoker (5 levels)', 'Control examination needed'),
 ('Drinker (3 levels)', 'Binge drinking'),
 ('Drinker (3 levels)', 'Grant'),
 ('Drinker (3 levels)', 'Marijuana use'),
 ('Drinker (3 levels)', 'French nationality'),
 ('Binge drinking', 'Other recreational drugs'),
 ('Marijuana use', 'Other recreational drugs'),
 ('Marijuana use', 'Eating junk food'),
 ('Marijuana use', 'Having only one parent'),
 ('Other recreational drugs', 'Informed about opportunities')]) 
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")
    inference = VariableElimination(model)
    hj=dict(df['Weight (kg)'])
    index = list(hj.keys())
    for i in range(len(df['Weight (kg)'])):
        if(df['Weight (kg)'][index[i]]>=38 and df['Weight (kg)'][index[i]]<=52):
            df['Weight (kg)'][index[i]]=1
        elif(df['Weight (kg)'][index[i]]>=53 and df['Weight (kg)'][index[i]]<=67):
            df['Weight (kg)'][index[i]]=2
        elif(df['Weight (kg)'][index[i]]>=68 and df['Weight (kg)'][index[i]]<=82):
            df['Weight (kg)'][index[i]]=3
        elif(df['Weight (kg)'][index[i]]>=83 and df['Weight (kg)'][index[i]]<=97):
            df['Weight (kg)'][index[i]]=4
        elif(df['Weight (kg)'][index[i]]>=98 and df['Weight (kg)'][index[i]]<=112):
            df['Weight (kg)'][index[i]]=5
        elif(df['Weight (kg)'][index[i]]>=113 and df['Weight (kg)'][index[i]]<=127):
            df['Weight (kg)'][index[i]]=6
        elif(df['Weight (kg)'][index[i]]>=128 and df['Weight (kg)'][index[i]]<=142):
            df['Weight (kg)'][index[i]]=7

    hj=dict(df['Systolic blood pressure (mmHg)'])
    index = list(hj.keys())
    for i in range(len(df['Systolic blood pressure (mmHg)'])):
        if(df['Systolic blood pressure (mmHg)'][index[i]]>=8 and df['Systolic blood pressure (mmHg)'][index[i]]<=11):
            df['Systolic blood pressure (mmHg)'][index[i]]=1
        elif(df['Systolic blood pressure (mmHg)'][index[i]]>11 and df['Systolic blood pressure (mmHg)'][index[i]]<=14):
            df['Systolic blood pressure (mmHg)'][index[i]]=2
        elif(df['Systolic blood pressure (mmHg)'][index[i]]>14 and df['Systolic blood pressure (mmHg)'][index[i]]<=17):
            df['Systolic blood pressure (mmHg)'][index[i]]=3 

    hj=dict(df['Diastolic blood pressure (mmHg)'])
    index = list(hj.keys())
    for i in range(len(df['Diastolic blood pressure (mmHg)'])):
        if(df['Diastolic blood pressure (mmHg)'][index[i]]>=4 and df['Diastolic blood pressure (mmHg)'][index[i]]<7):
            df['Diastolic blood pressure (mmHg)'][index[i]]=1
        elif(df['Diastolic blood pressure (mmHg)'][index[i]]>=7 and df['Diastolic blood pressure (mmHg)'][index[i]]<10):
            df['Diastolic blood pressure (mmHg)'][index[i]]=2
        elif(df['Diastolic blood pressure (mmHg)'][index[i]]>=10 and df['Diastolic blood pressure (mmHg)'][index[i]]<=13):
            df['Diastolic blood pressure (mmHg)'][index[i]]=3

    hj=dict(df['Heart rate (bpm)'])
    index = list(hj.keys())
    for i in range(len(df['Heart rate (bpm)'])):
        if(df['Heart rate (bpm)'][index[i]]>=40 and df['Heart rate (bpm)'][index[i]]<=54):
            df['Heart rate (bpm)'][index[i]]=1
        elif(df['Heart rate (bpm)'][index[i]]>=55 and df['Heart rate (bpm)'][index[i]]<=69):
            df['Heart rate (bpm)'][index[i]]=2
        elif(df['Heart rate (bpm)'][index[i]]>=70 and df['Heart rate (bpm)'][index[i]]<=84):
            df['Heart rate (bpm)'][index[i]]=3
        elif(df['Heart rate (bpm)'][index[i]]>=85 and df['Heart rate (bpm)'][index[i]]<=99):
            df['Heart rate (bpm)'][index[i]]=4
        elif(df['Heart rate (bpm)'][index[i]]>=100 and df['Heart rate (bpm)'][index[i]]<=114):
            df['Heart rate (bpm)'][index[i]]=5
        elif(df['Heart rate (bpm)'][index[i]]>=115 and df['Heart rate (bpm)'][index[i]]<=129):
            df['Heart rate (bpm)'][index[i]]=6

    

        
    q = inference.query(['Anxiety symptoms'],evidence={
                                                      'Age (4 levels)': int(dic['age']),
                                                      'Gender': int(dic['Gender']) ,
                                                      'French nationality': int(dic['french_nationality']) ,             
                                                      'Field of study': int(dic['study']) ,
                                                      'Year of university': int(dic['year']) ,
                                                      'Learning disabilities': int(dic['l_disability']) ,
                                                      'Difficulty memorizing lessons': int(dic['d_mem_lessons']) ,
                                                      'Professional objective': int(dic['p_objective']) ,
                                                      'Informed about opportunities': int(dic['opportunities']) ,
                                                      'Satisfied with living conditions':int(dic['l_conditions']) ,
                                                      'Living with a partner/child': int(dic['l_partner']) ,
                                                      'Parental home': int(dic['p_home']) ,
                                                      'Having only one parent': int(dic['one_parent']) ,
                                                      'At least one parent unemployed': int(dic['p_unemployed']) ,
                                                      'Siblings': int(dic['siblings']) ,
                                                      'Long commute': int(dic['l_commute']) ,
                                                      'Mode of transportation': int(dic['m_transport']) ,
                                                      'Financial difficulties': int(dic['f_difficulties']) ,
                                                      'Grant': int(dic['grant']) ,
                                                      'Additional income': int(dic['a_income'])  ,
                                                      'Public health insurance ': int(dic['public_insurance']) ,
                                                      'Private health insurance ': int(dic['private_insurance']) ,
                                                      'C.M.U.': int(dic['cmu']) ,
                                                      'Irregular rhythm of meals': int(dic['i_meals']) ,
                                                      'Unbalanced meals': int(dic['u_meals']) ,
                                                      'Eating junk food': int(dic['junk_food']) ,
                                                      'On a diet':  int(dic['diet']) ,
                                                      'Irregular rhythm or unbalanced meals': int(dic['i_u_meals']) ,
                                                      'Physical activity(3 levels)': int(dic['p_activity']) ,
                                                      'Weight (kg)': int(dic['weight']) ,
                                                      'Height (cm)': 1,
                                                      'Overweight and obesity':  int(dic['overweight']) ,
                                                      'Systolic blood pressure (mmHg)': int(dic['s_bp']) ,
                                                      'Diastolic blood pressure (mmHg)': 1 ,
                                                      'Prehypertension or hypertension': int(dic['hypertension']) ,
                                                      'Heart rate (bpm)': int(dic['heartrate']) ,
                                                      'Abnormal heart rate': int(dic['a_heartrate']) ,
                                                      'Distant visual acuity of right eye (score /10)': int(dic['act_1']) ,
                                                      'Distant visual acuity of left eye (score /10)': int(dic['act_2']) ,
                                                      'Close visual acuity of right eye (score /10)': int(dic['act_3']) ,
                                                      'Close visual acuity of left eye (score /10)': int(dic['act_4']) ,
                                                      'Decreased in distant visual acuity': int(dic['act_5']) ,
                                                      'Decreased in close visual acuity': int(dic['act_6']) ,
                                                      'Urinalysis (hematuria)': int(dic['u_h']) ,
                                                      'Urinalysis leukocyturia)': int(dic['u_l']) ,
                                                      'Abnormal urinalysis': int(dic['abnormal_u']) ,
                                                      'Vaccination up to date': int(dic['vaccine']) ,
                                                      'Control examination needed': int(dic['control_exam']) ,
                                                      'Panic attack symptoms': int(dic['panic_atck']) ,
                                                      'Cigarette smoker (5 levels)': int(dic['c_smoker']) ,
                                                      'Drinker (3 levels)': int(dic['drinker']) ,
                                                      'Binge drinking': int(dic['b_drinking']) ,
                                                      'Marijuana use': int(dic['marjiuana']) ,
                                                      'Other recreational drugs':  int(dic['o_drugs'])      
                                            }, joint=False)['Anxiety symptoms']
    return q.values[1]

