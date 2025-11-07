from langchain_core.prompts import PromptTemplate

template = """Вы – опытный наставник по программированию. 
Всегда отвечайте с пояснением и приведением примера кода.

Примеры:
Вопрос: {example_question1}
Ответ: {example_answer1}

Вопрос: {example_question2}
Ответ: {example_answer2}

Теперь ответь на вопрос пользователя аналогичным образом, **с пояснениями и примером кода**.
Вопрос: {user_question}
Ответ:"""

# Заполняем примеры
prompt = PromptTemplate(
    input_variables=["example_question1", "example_answer1",
                     "example_question2", "example_answer2", "user_question"],
    template=template
)

# Два примера:
ex_q1 = "Как вычислить факториал числа рекурсивно на Python?"
ex_a1 = "Чтобы вычислить факториал рекурсивно, можно определить функцию, \
которая вызывает себя с уменьшенным аргументом. Например:\n\
```python\n\
def factorial(n):\n\
    if n <= 1:\n\
        return 1\n\
    else:\n\
        return n * factorial(n-1)\n\
```\n\
Здесь функция вызывает себя, уменьшая n, пока не достигнет 1."

ex_q2 = "Как в Java прочитать файл построчно?"
ex_a2 = "В Java для построчного чтения файла можно использовать BufferedReader. Например:\n\
```java\n\
try (BufferedReader br = new BufferedReader(new FileReader(\"file.txt\"))) {\n\
    String line;\n\
    while ((line = br.readLine()) != null) {\n\
        System.out.println(line);\n\
    }\n\
} catch (IOException e) {\n\
    e.printStackTrace();\n\
}\n\
```\n\
Этот код открывает файл и выводит его содержимое на экран построчно."

# Формируем окончательный промпт для нового вопроса
user_q = "Как отсортировать список объектов по полю в Python?"
final_prompt = prompt.format(
    example_question1=ex_q1, example_answer1=ex_a1,
    example_question2=ex_q2, example_answer2=ex_a2,
    user_question=user_q
)
print(final_prompt)
