from pydantic import BaseModel, Field
from typing import List, Optional
import json

# 1. 定义一个完整的用户管理系统模型
class Address(BaseModel):
    """用户的地址信息"""
street: str = Field(description="街道地址", json_schema_extra={"example": "科技园路123号"})
city: str = Field(description="城市", json_schema_extra={"example": "北京市"})
postal_code: str = Field(description="邮政编码", json_schema_extra={"example": "100000"})
class User(BaseModel):
    """
    用户信息模型

    这个模型用于存储系统用户的基本信息、联系方式和元数据。
    """
    id: int = Field(description="用户唯一标识符", example=1001)
    username: str = Field(description="用户名，3-20个字符", min_length=3, max_length=20, example="john_doe")
    email: str = Field(description="邮箱地址", example="john@example.com")
    age: Optional[int] = Field(None, description="用户年龄，范围18-120", ge=18, le=120, example=25)
    tags: List[str] = Field(default=[], description="用户标签列表", example=["vip", "early_adopter"])
    address: Optional[Address] = Field(None, description="用户地址信息")
    is_active: bool = Field(True, description="账户是否激活")

class UserResponse(BaseModel):
    """API用户响应模型"""
    success: bool = Field(description="请求是否成功")
    data: User = Field(description="用户数据")
    message: str = Field(description="响应消息", example="用户信息获取成功")

# 2. 展示各种文档生成方式
def demonstrate_documentation():
    print("=" * 60)
    print("📚 Pydantic文档自动生成演示")
    print("=" * 60)

    # 方式1：直接查看模型的docstring
    print("\n1. 📖 模型类文档字符串:")
    print(User.__doc__)

    # 方式2：生成JSON Schema（最常用）
    print("\n2. 📋 完整的JSON Schema:")
    schema = User.model_json_schema()
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    print("\n3. 🎯 简化的Schema（用于前端）：")
    simplified_schema = json.dumps(User.model_json_schema(), indent=2)
    print(simplified_schema)

    # 方式3：获取字段信息
    print("\n4. 🔍 字段详细信息:")
    for field_name, field_info in User.model_fields.items():
        print(f"  {field_name}:")
        print(f"    类型: {field_info.type_}")
        print(f"    必需: {not field_info.required}")
        if field_info.field_info.description:
            print(f"    描述: {field_info.field_info.description}")
        if field_info.field_info.example:
            print(f"    示例: {field_info.field_info.example}")
        print()

    # 方式4：生成Markdown文档
    print("\n5. 📝 Markdown格式文档:")
    generate_markdown_docs()

def generate_markdown_docs():
    """生成Markdown格式的API文档"""
    markdown = f"""
# User API 文档

## User 模型

{User.__doc__}

### 字段说明

| 字段名 | 类型 | 必需 | 描述 | 示例 |
|--------|------|------|------|------|"""

    for field_name, field_info in User.__fields__.items():
        required = "是" if field_info.required else "否"
        field_type = str(field_info.type_).replace("typing.", "")
        description = field_info.field_info.description or "无描述"
        example = field_info.field_info.example or "无示例"

        markdown += f"\n| {field_name} | {field_type} | {required} | {description} | {example} |"

    print(markdown)

# 3. 实际应用：生成API文档
def generate_api_documentation():
    """生成完整的API文档"""
    print("\n" + "=" * 60)
    print("🌐 API接口文档生成")
    print("=" * 60)

    # 生成请求/响应示例
    example_user = User(
        id=1001,
        username="demo_user",
        email="demo@example.com",
        age=28,
        tags=["developer", "beta_tester"],
        address=Address(
            street="创新路456号",
            city="上海市",
            postal_code="200000"
        )
    )

    example_response = UserResponse(
        success=True,
        data=example_user,
        message="用户信息获取成功"
    )

    print("\n📨 请求示例:")
    print("GET /api/users/1001")

    print("\n📬 响应示例:")
    print(json.dumps(example_response.dict(), indent=2, ensure_ascii=False))

    print("\n🔧 响应模型Schema:")
    response_schema = UserResponse.schema()
    print(json.dumps(response_schema, indent=2, ensure_ascii=False))

# 4. 高级功能：动态文档生成
def dynamic_documentation():
    """动态生成文档"""
    print("\n" + "=" * 60)
    print("🔄 动态文档功能")
    print("=" * 60)

    # 获取所有模型的引用
    models = [User, Address, UserResponse]

    for model in models:
        print(f"\n📦 模型: {model.__name__}")
        schema = model.schema()

        print(f"   描述: {schema.get('description', '无描述')}")
        print(f"   字段数: {len(schema['properties'])}")

        required_fields = schema.get('required', [])
        print(f"   必需字段: {required_fields}")

if __name__ == "__main__":
    # 运行所有演示
    demonstrate_documentation()
    generate_api_documentation()
    dynamic_documentation()

    # 额外：在LangChain中的应用示例
    print("\n" + "=" * 60)
    print("🤖 在LangChain结构化输出中的应用")
    print("=" * 60)

    # LangChain可以利用这些schema来指导AI输出
    print("LangChain可以使用这个JSON Schema来约束AI的输出格式：")
    structured_schema = User.schema()
    print(f"Schema可以直接传递给LangChain的structured_output方法")
    print(f"AI将会按照这个格式生成响应，包含：")
    print(f"  - {len(structured_schema['properties'])} 个预定义字段")
    print(f"  - 明确的类型检查和验证规则")