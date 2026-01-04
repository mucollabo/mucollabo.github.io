---
classes: wide

title: "FastAPI 시작해봅시다"

excerpt: "FastAPI를 사용하여 웹 애플리케이션을 만들어보세요 😘"

categories:
  - FastAPI

tags:
  - FastAPI
  - Rest API
  - web framework
  - Python
  - uvicorn
  - pydantic
  - Depends(Dependency Injection)
  - Swagger
  - OpenAPI
  - Redoc

comments: true
---

### FastAPI 를 접하게 된 나의 계기와, 간단한 FastAPI 소개

## Why FastAPI?

현(2026년 1월 4일) 기준으로 나는 약 1년 전(2025년 1월) 서비스 [뮤콜라보](https://mucollabo.com)를 개발하기로 하였다.
당시 나는 Node.js 와 Express.js 를 사용하여 REST API 서버를 개발하는 백엔드 개발자였다.
그리고 데이터분석을 공부하면서 Python 에 익숙했었고, 데이터분석은 앞으로 개발하는 서비스에 제외할 수 없는 소금과 같은 존재라는 것을 직감하고 있었다.
이왕 데이터 분석을 할거라면, Python 기반의 웹 프레임워크를 사용하여 REST API 서버를 개발하는 것이 더 효율적일것이라고 생각했다.
하지만 Django 나 Flask 같은 기존의 Python 웹 프레임워크는 Node.js Express.js 와 같은 비동기 처리에 최적화 되어 있지 않았다.
그래서 고민하던 중에 FastAPI 라는 Python 기반의 비동기 웹 프레임워크를 알게 되었고, 이를 사용하여 REST API 서버를 개발하기로 결심했다.
프론트는 모바일앱을 개발하는 것이였기 때문에 React Native 를 사용하기로 하였다. React Native 이야기는 React Native 메뉴에서 다루기로 한다.

## FastAPI 란?

FastAPI 는 Sebastian Ramírez 가 개발한 Python 기반의 비동기 웹 프레임워크이다.
FastAPI 와 Sebastian 에 관해 궁금한 것이 있다면, 2025 년 12월에 있었던 [FastAPI mini documentary](https://youtu.be/mpR8ngthqiE?si=D5H7Ckfx3339-AQj) 영상을 참고하길 바란다.
Sebastian 에게 감사와 존경을 표한다.

내가 생각하는 FastAPI 의 장점은 다음과 같다.

1. **비동기화(Asynchronous Support)**: 비동기 프로그래밍을 지원하여 높은 처리량을 제공합니다.
2. **빠른성능(High Performance)**: Starlette 기반으로 빠른 성능을 제공합니다.
3. **의존성주입(Dependency Injection)**: 의존성 주입(Dependency Injection) 기능을 통해 코드 재사용성을 높입니다.
4. **유효성검증(Pydantic Integration)**: Pydantic을 사용하여 데이터 유효성 검증과 자동 문서화를 지원합니다.
5. **API 자동문서화(Automatic Documentation)**: Swagger UI와 Redoc을 자동으로 제공하여 API 문서화가 간편합니다.
6. **타입힌트(Type Hints)**: Python의 타입 힌트를 사용하여 자동으로 API 문서를 생성하고, 입력값 검증을 수행합니다.

{% if page.comments != false %}
{% include disqus.html %}
{% endif %}
