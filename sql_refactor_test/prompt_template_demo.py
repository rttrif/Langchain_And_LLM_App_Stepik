import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()


def demo_template_rendering():
    """Демонстрация рендеринга шаблона с разными параметрами."""
    print("=== Демонстрация рендеринга шаблона ===\n")

    # Создание PromptTemplate с явным указанием input_variables
    template_path = "sql_refactor_template.jinja2"
    with open(template_path, "r", encoding="utf-8") as file:
        template_string = file.read()

    # Для Способа 2 используем эти параметры:
    prompt_template = PromptTemplate(
        template=template_string,
        input_variables=["source_sql", "target_dialect", "keyword_style", "comments_instruction"]
    )

    # Пример 1: Конвертация из MSSQL в PostgreSQL
    example1_params = {
        "source_sql": "SELECT TOP 10 * FROM Employees WHERE DepartmentID = 1",
        "target_dialect": "PostgreSQL",
        "keyword_style": "UPPERCASE",
        "comments_instruction": "Add brief inline comments to explain key steps or complex conditions."
    }
    prompt1 = prompt_template.invoke(example1_params)
    print("Пример 1 - MSSQL -> PostgreSQL:")
    print(prompt1.text)
    print("\n" + "="*50 + "\n")

    # Пример 2: Форматирование сложного запроса
    example2_params = {
        "source_sql": """SELECT e.name, d.department_name, COUNT(p.project_id) as project_count 
                        FROM employees e 
                        INNER JOIN departments d ON e.dept_id = d.dept_id 
                        LEFT JOIN projects p ON e.emp_id = p.lead_emp_id 
                        WHERE e.hire_date > '2023-01-01' 
                        GROUP BY e.name, d.department_name 
                        HAVING COUNT(p.project_id) > 5""",
        "target_dialect": "MySQL",
        "keyword_style": "lowercase",
        "comments_instruction": "No comments are needed."
    }
    prompt2 = prompt_template.invoke(example2_params)
    print("Пример 2 - Сложный запрос для MySQL:")
    print(prompt2.text)
    print("\n" + "="*50 + "\n")


def demo_with_llm():
    """Опционально: вызов LLM для получения готового SQL."""
    print("=== Демонстрация с вызовом LLM ===\n")

    # Создание цепи LangChain
    template_path = "sql_refactor_template.jinja2"
    with open(template_path, "r", encoding="utf-8") as file:
        template_string = file.read()

    prompt_template = PromptTemplate(
        template=template_string,
        input_variables=["source_sql", "target_dialect", "keyword_style", "comments_instruction"]
    )

    llm  = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.0)

    # Параметры для запроса
    test_params = {
        "source_sql": "SELECT TOP 5 * FROM Users WHERE Active = 1",
        "target_dialect": "PostgreSQL",
        "keyword_style": "UPPERCASE",
        "comments_instruction": "Add brief inline comments to explain key steps or complex conditions."
    }

    # Формирование и выполнение запроса
    chain = prompt_template | llm
    try:
        response = chain.invoke(test_params)
        print("Рефакторенный SQL:")
        print(response.content)
    except Exception as e:
        print(f"Ошибка при вызове LLM: {e}")


if __name__ == "__main__":
    demo_template_rendering()
    demo_with_llm()