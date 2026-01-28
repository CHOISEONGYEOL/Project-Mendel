"""
Spiral Engine - Flask REST API
웹 서비스용 API 엔드포인트

실행: flask --app api run --debug
또는: python api.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from spiral_engine import (
    PedigreeGenerator, DNATableGenerator,
    PedigreeVisualizer, LogicValidator,
    InheritanceMode
)

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 전역 객체
generator = PedigreeGenerator()
table_generator = DNATableGenerator()
visualizer = PedigreeVisualizer()
validator = LogicValidator()


@app.route('/')
def index():
    """API 정보"""
    return jsonify({
        'name': 'Spiral Engine API',
        'version': '1.0.0',
        'description': '수능 생명과학 유전 문제 생성 API',
        'endpoints': {
            '/generate': 'POST - 새 문제 생성',
            '/modes': 'GET - 사용 가능한 유전 방식 목록',
            '/validate': 'POST - 문제 검증'
        }
    })


@app.route('/modes', methods=['GET'])
def get_modes():
    """사용 가능한 유전 방식 목록"""
    modes = [
        {'id': 'autosomal_recessive', 'name': '상염색체 열성'},
        {'id': 'autosomal_dominant', 'name': '상염색체 우성'},
        {'id': 'x_linked_recessive', 'name': 'X염색체 연관 열성'},
        {'id': 'x_linked_dominant', 'name': 'X염색체 연관 우성'},
    ]
    return jsonify({'modes': modes})


@app.route('/generate', methods=['POST'])
def generate_problem():
    """
    새 유전 문제 생성

    Request Body:
    {
        "mode": "autosomal_recessive",  // 유전 방식
        "num_traits": 1,                 // 형질 수 (1 또는 2)
        "difficulty": "medium",          // 난이도 (easy/medium/hard)
        "seed": null                     // 랜덤 시드 (선택)
    }
    """
    try:
        data = request.get_json() or {}

        # 파라미터 추출
        mode_str = data.get('mode', 'autosomal_recessive')
        num_traits = data.get('num_traits', 1)
        difficulty = data.get('difficulty', 'medium')
        seed = data.get('seed')

        # 유전 방식 매핑
        mode_map = {
            'autosomal_recessive': InheritanceMode.AUTOSOMAL_RECESSIVE,
            'autosomal_dominant': InheritanceMode.AUTOSOMAL_DOMINANT,
            'x_linked_recessive': InheritanceMode.X_LINKED_RECESSIVE,
            'x_linked_dominant': InheritanceMode.X_LINKED_DOMINANT,
        }
        mode = mode_map.get(mode_str, InheritanceMode.AUTOSOMAL_RECESSIVE)

        # 생성기 초기화 (시드 포함)
        gen = PedigreeGenerator(seed=seed)

        # 문제 생성
        if num_traits == 1:
            success, family = gen.generate_complete_problem(mode=mode)
        else:
            modes = [mode]
            if mode in [InheritanceMode.AUTOSOMAL_RECESSIVE,
                       InheritanceMode.AUTOSOMAL_DOMINANT]:
                modes.append(InheritanceMode.X_LINKED_RECESSIVE)
            else:
                modes.append(InheritanceMode.AUTOSOMAL_RECESSIVE)
            success, family = gen.generate_multi_trait_problem(
                num_traits=num_traits,
                modes=modes[:num_traits]
            )

        if not success:
            return jsonify({
                'success': False,
                'error': '문제 생성 실패'
            }), 500

        genes = family.genes

        # DNA 상대량 표 생성
        table = table_generator.generate_table(
            family, genes,
            num_persons=4,
            difficulty=difficulty
        )

        # 검증
        validation = validator.validate_logic(family, genes)

        if not validation.is_valid:
            return jsonify({
                'success': False,
                'error': '논리 검증 실패',
                'validation_errors': [
                    {'message': e.message, 'details': e.details}
                    for e in validation.get_errors()
                ]
            }), 500

        # 문제 데이터 구성
        question_data = table_generator.create_question_data(
            table, family, genes
        )

        # 이미지 생성
        gene_symbol = genes[0].symbol if genes else None
        problem_img = visualizer.create_problem_image(
            family, gene_symbol, hide_genotypes=True
        )
        answer_img = visualizer.create_answer_image(
            family, gene_symbol
        )

        # 가족 정보 구성
        family_data = []
        for person in family.all_members:
            member_data = {
                'id': person.id,
                'display_name': person.display_name,
                'gender': person.gender.value,
                'generation': person.generation,
                'traits': {}
            }
            for symbol, trait in person.traits.items():
                member_data['traits'][symbol] = {
                    'genotype': trait.genotype,
                    'phenotype': trait.phenotype.value,
                    'is_carrier': trait.is_carrier,
                    'dna_amounts': trait.get_dna_amounts()
                }
            family_data.append(member_data)

        return jsonify({
            'success': True,
            'config': {
                'mode': mode.value,
                'num_traits': num_traits,
                'difficulty': difficulty
            },
            'family': family_data,
            'genes': [
                {
                    'symbol': g.symbol,
                    'dominant_allele': g.dominant_allele,
                    'recessive_allele': g.recessive_allele,
                    'chromosome_type': g.chromosome_type.value
                }
                for g in genes
            ],
            'dna_table': question_data,
            'images': {
                'problem': f"data:image/png;base64,{problem_img}",
                'answer': f"data:image/png;base64,{answer_img}"
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/validate', methods=['POST'])
def validate_problem():
    """
    사용자 제출 문제 검증

    Request Body:
    {
        "family": [...],  // 가족 데이터
        "genes": [...]    // 유전자 데이터
    }
    """
    # TODO: 사용자 제출 데이터 검증 구현
    return jsonify({
        'success': True,
        'message': '검증 API는 추후 구현 예정'
    })


if __name__ == '__main__':
    print("=" * 50)
    print("Spiral Engine API Server")
    print("=" * 50)
    print("Server starting at http://localhost:5000")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
