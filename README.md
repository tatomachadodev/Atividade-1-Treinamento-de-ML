# Anamnese Alzheimer

Aplicação acadêmica de triagem de risco de Alzheimer usando Machine Learning. O sistema combina uma API em FastAPI, um modelo de árvore de decisão treinado com Scikit-learn e uma interface web para preenchimento da anamnese.

## 1. Dataset e problema

O dataset escolhido é o `Alzheimer's Disease Dataset`, que reúne informações de pacientes relacionadas a aspectos demográficos, hábitos de vida, condições de saúde, indicadores clínicos e sinais cognitivos/comportamentais.

O problema representado é uma **classificação binária**: estimar, a partir dessas informações, se o registro do paciente está associado ou não ao diagnóstico de Alzheimer (`Diagnosis`). O dataset possui 2.149 registros, sendo 1.389 da classe `0` e 760 da classe `1`.

Durante o treinamento, as colunas `PatientID` e `DoctorInCharge` são removidas por não contribuírem como características clínicas do paciente.

## 2. Variável-alvo e classes

A variável-alvo é:

- `Diagnosis`: resultado que o modelo deve prever.

As classes possíveis são:

- `0`: não possui diagnóstico de Alzheimer no registro;
- `1`: possui diagnóstico de Alzheimer no registro.

O modelo é treinado com uma `DecisionTreeClassifier`, usando 80% dos dados para treinamento e 20% para teste.

## 3. Informações usadas como entrada

A aplicação utiliza os seguintes dados como entrada do modelo:

- **Perfil:** idade, gênero, etnia, escolaridade, IMC e tabagismo;
- **Histórico e hábitos:** histórico familiar de Alzheimer, consumo de álcool, atividade física, qualidade da dieta e qualidade do sono;
- **Condições de saúde:** doença cardiovascular, diabetes, depressão, lesão cerebral e hipertensão;
- **Indicadores clínicos:** pressão arterial sistólica e diastólica, colesterol total, LDL, HDL e triglicerídeos;
- **Cognição e funcionalidade:** pontuação MMSE, avaliação funcional, atividades da vida diária (ADL), queixas de memória, problemas comportamentais, confusão, desorientação, alterações de personalidade, dificuldade para completar tarefas e esquecimento.

Os dados são validados pela API antes da previsão. Por exemplo, a idade deve estar entre 60 e 90 anos e a pressão sistólica deve ser maior que a diastólica.

## 4. Usuários e finalidade

A solução pode ser utilizada, em contexto acadêmico e experimental, por:

- estudantes e professores para demonstrar um fluxo completo de treinamento e uso de um modelo de classificação;
- profissionais de saúde como apoio inicial à organização de informações e triagem;
- pesquisadores interessados em testar uma aplicação de predição baseada em dados clínicos.

A finalidade é gerar uma estimativa de risco a partir dos dados informados. O resultado não substitui consulta, avaliação clínica ou diagnóstico realizado por um profissional habilitado.

## 5. O que acontece após a classificação

A API calcula a probabilidade estimada da classe `1` e converte essa probabilidade em um nível de risco:

| Probabilidade estimada | Nível | Orientação apresentada |
| --- | --- | --- |
| menor que 20% | **BAIXO** | manter hábitos saudáveis e monitoramento ocasional |
| de 20% a menos de 40% | **MODERADO** | monitoramento regular e hábitos saudáveis |
| de 40% a menos de 60% | **ALTO** | acompanhamento médico próximo |
| 60% ou mais | **CRÍTICO** | avaliação médica imediata |

A aplicação exibe o nível de risco, a probabilidade estimada e uma recomendação correspondente. Essas faixas são regras de apresentação da aplicação e não representam, sozinhas, um diagnóstico médico.

## 6. Interface e experiência de uso

A interface web é uma anamnese dividida em quatro partes:

1. perfil do paciente;
2. hábitos e bem-estar;
3. indicadores clínicos;
4. cognição e funcionalidade.

O usuário preenche campos numéricos, listas de seleção e escalas de avaliação. Ao selecionar **Analisar risco**, os dados são enviados para a API, que retorna a classificação. O resultado aparece na própria página com um indicador visual do nível de risco, a porcentagem estimada e o próximo passo recomendado. Também é possível iniciar uma nova avaliação.

## 7. Como executar

### Instalar dependências

```bash
pip install fastapi uvicorn pandas scikit-learn joblib
```

### Iniciar a API

Na raiz do projeto, execute:

```bash
python app.py
```

A API ficará disponível em `http://127.0.0.1:8000`. A documentação interativa pode ser acessada em `http://127.0.0.1:8000/docs`.

### Abrir a interface

Com a API em execução, abra `anamnese_alzheimer/index.html` no navegador. Se o navegador bloquear requisições ao abrir o arquivo diretamente, sirva a pasta com um servidor local, por exemplo:

```bash
python -m http.server 5500 --directory anamnese_alzheimer
```

Depois, acesse `http://127.0.0.1:5500`.

### Treinar novamente o modelo

Para gerar novamente `alzheimers_model.pkl` a partir do CSV:

```bash
python train_model.py
```
