import pandas as pd
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense

demographic_cols = [
    'ncodpers', 'fecha_alta', 'ind_empleado', 'pais_residencia', 'sexo', 'age',
    'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes',
    'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom',
    'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento'
]

notuse = ["ult_fec_cli_1t", "nomprov", 'fecha_dato']

product_col = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

df_train = pd.read_csv('dataset/train_ver2.csv')
df_test = pd.read_csv('dataset/test_ver2.csv')


def filter_data(df):
    df = df[df['ind_nuevo'] == 0]
    df = df[df['antiguedad'] != -999999]
    df = df[df['indrel'] == 1]
    df = df[df['indresi'] == 'S']
    df = df[df['indfall'] == 'N']
    df = df[df['tipodom'] == 1]
    df = df[df['ind_empleado'] == 'N']
    df = df[df['pais_residencia'] == 'ES']
    df = df[df['indrel_1mes'] == 1]
    df = df[df['tiprel_1mes'] == ('A' or 'I')]
    df = df[df['indext'] == 'N']


filter_data(df_train)

drop_column = [
    'ind_nuevo', 'indrel', 'indresi', 'indfall', 'tipodom', 'ind_empleado',
    'pais_residencia', 'indrel_1mes', 'indext', 'conyuemp', 'fecha_alta',
    'tiprel_1mes'
]

df_train.drop(drop_column, axis=1, inplace=True)
df_test.drop(drop_column, axis=1, inplace=True)

df_test["renta"] = pd.to_numeric(df_test["renta"], errors="coerce")
unique_prov = df_test[df_test.cod_prov.notnull()].cod_prov.unique()
grouped = df_test.groupby("cod_prov")["renta"].median()


def impute_renta(df):
    df["renta"] = pd.to_numeric(df["renta"], errors="coerce")
    for cod in unique_prov:
        df.loc[df['cod_prov'] == cod,
               ['renta']] = df.loc[df['cod_prov'] == cod, ['renta']].fillna({
                   'renta':
                   grouped[cod]
               }).values
    df.renta.fillna(df_test["renta"].median(), inplace=True)


impute_renta(df_train)
impute_renta(df_test)


def drop_na(df):
    df.dropna(axis=0, subset=['ind_actividad_cliente'], inplace=True)


drop_na(df_train)

# These column are categories feature, I'll transform them using get_dummy
dummy_col = ['sexo', 'canal_entrada', 'cod_prov', 'segmento']
dummy_col_select = ['canal_entrada', 'cod_prov']

limit = int(0.01 * len(df_train.index))
use_dummy_col = {}

for col in dummy_col_select:
    trainlist = df_train[col].value_counts()
    use_dummy_col[col] = []
    for i, item in enumerate(trainlist):
        if item > limit:
            use_dummy_col[col].append(df_train[col].value_counts().index[i])


def get_dummy(df):
    for col in dummy_col_select:
        for item in df[col].unique():
            if item not in use_dummy_col[col]:
                row_index = df[col] == item
                df.loc[row_index, col] = np.nan
    return pd.get_dummies(df, prefix=dummy_col, columns=dummy_col)


df_train = get_dummy(df_train)
df_test = get_dummy(df_test)

df["age"] = pd.to_numeric(df["age"], errors="coerce")
max_age = 80

df["age"] = df['age'].apply(lambda x: min(x, max_age))
df["age"] = df['age'].apply(lambda x: round(x / max_age, 6))


def clean_renta(df):
    max_renta = 1.0e6

    df["renta"] = df['renta'].apply(lambda x: min(x, max_renta))
    df["renta"] = df['renta'].apply(lambda x: round(x / max_renta, 6))


def clean_antigue(df):
    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    df["antiguedad"] = df["antiguedad"].replace(-999999,
                                                df['antiguedad'].median())
    max_antigue = 256

    df["antiguedad"] = df['antiguedad'].apply(lambda x: min(x, max_antigue))
    df["antiguedad"] = df['antiguedad'].apply(
        lambda x: round(x / max_antigue, 6))


clean_age(df_train)
clean_age(df_test)

clean_renta(df_train)
clean_renta(df_test)

clean_antigue(df_train)
clean_antigue(df_test)

product_col_5 = [col for col in df_train.columns if '_ult1_5' in col]
product_col_4 = [col for col in df_train.columns if '_ult1_4' in col]
product_col_3 = [col for col in df_train.columns if '_ult1_3' in col]
product_col_2 = [col for col in df_train.columns if '_ult1_2' in col]
product_col_1 = [col for col in df_train.columns if '_ult1_1' in col]

df_train['tot5'] = df_train[product_col_5].sum(axis=1)
df_test['tot5'] = df_test[product_col_5].sum(axis=1)
df_train['tot4'] = df_train[product_col_4].sum(axis=1)
df_test['tot4'] = df_test[product_col_4].sum(axis=1)
df_train['tot3'] = df_train[product_col_3].sum(axis=1)
df_test['tot3'] = df_test[product_col_3].sum(axis=1)
df_train['tot2'] = df_train[product_col_2].sum(axis=1)
df_test['tot2'] = df_test[product_col_2].sum(axis=1)
df_train['tot1'] = df_train[product_col_1].sum(axis=1)
df_test['tot1'] = df_test[product_col_1].sum(axis=1)

for col in product_col[2:]:
    df_train[col + '_past'] = (df_train[col + '_5'] + df_train[col + '_4'] +
                               df_train[col + '_3'] + df_train[col + '_2'] +
                               df_train[col + '_1']) / 5
    df_test[col + '_past'] = (df_test[col + '_5'] + df_test[col + '_4'] +
                              df_test[col + '_3'] + df_test[col + '_2'] +
                              df_test[col + '_1']) / 5

for pro in product_col[2:]:
    df_train[pro +
             '_past'] = df_train[pro + '_past'] * (1 - df_train[pro + '_5'])
    df_test[pro + '_past'] = df_test[pro + '_past'] * (1 - df_test[pro + '_5'])

for col in product_col[2:]:
    for month in range(2, 6):
        df_train[col + '_' + str(month) +
                 '_diff'] = df_train[col + '_' +
                                     str(month)] - df_train[col + '_' +
                                                            str(month - 1)]
        df_test[col + '_' + str(month) +
                '_diff'] = df_test[col + '_' +
                                   str(month)] - df_test[col + '_' +
                                                         str(month - 1)]
        df_train[col + '_' + str(month) +
                 '_add'] = df_train[col + '_' + str(month) +
                                    '_diff'].apply(lambda x: max(x, 0))
        df_test[col + '_' + str(month) +
                '_add'] = df_test[col + '_' + str(month) +
                                  '_diff'].apply(lambda x: max(x, 0))

product_col_5_diff = [col for col in df_train.columns if '5_diff' in col]
product_col_4_diff = [col for col in df_train.columns if '4_diff' in col]
product_col_3_diff = [col for col in df_train.columns if '3_diff' in col]
product_col_2_diff = [col for col in df_train.columns if '2_diff' in col]

product_col_5_add = [col for col in df_train.columns if '5_add' in col]
product_col_4_add = [col for col in df_train.columns if '4_add' in col]
product_col_3_add = [col for col in df_train.columns if '3_add' in col]
product_col_2_add = [col for col in df_train.columns if '2_add' in col]

product_col_all_diff = [col for col in df_train.columns if '_diff' in col]
product_col_all_add = [col for col in df_train.columns if '_add' in col]

df_train['tot5_add'] = df_train[product_col_5_add].sum(axis=1)
df_test['tot5_add'] = df_test[product_col_5_add].sum(axis=1)
df_train['tot4_add'] = df_train[product_col_4_add].sum(axis=1)
df_test['tot4_add'] = df_test[product_col_4_add].sum(axis=1)
df_train['tot3_add'] = df_train[product_col_3_add].sum(axis=1)
df_test['tot3_add'] = df_test[product_col_3_add].sum(axis=1)
df_train['tot2_add'] = df_train[product_col_2_add].sum(axis=1)
df_test['tot2_add'] = df_test[product_col_2_add].sum(axis=1)

print(df_train.head())
print(df_test.head())

cols = list(
    df_train.drop(['target', 'ncodpers'] + product_col_all_diff +
                  product_col_all_add, 1).columns.values)

id_preds = defaultdict(list)
ids = df_test['ncodpers'].values

# predict model
y_train = pd.get_dummies(df_train['target'].astype(int))
x_train = df_train[cols]

# create model
model = Sequential()
model.add(Dense(150, input_dim=len(cols), activation='relu'))
model.add(Dense(22, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['categorical_accuracy'])

model.fit(x_train.as_matrix(),
          y_train.as_matrix(),
          validation_split=0.2,
          nb_epoch=150,
          batch_size=10)
#model.fit(x_train.as_matrix(), y_train.as_matrix(), nb_epoch=150, batch_size=10)

x_test = df_test[cols]
x_test = x_test.fillna(0)

p_test = model.predict(x_test.as_matrix())

for id, p in zip(ids, p_test):
    #id_preds[id] = list(p)
    id_preds[id] = [0, 0] + list(p)
