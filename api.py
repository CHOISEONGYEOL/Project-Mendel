"""
api.py - Spiral Engine REST API
Flask 기반 웹 API 서버
"""

from flask import Flask, jsonify, request, render_template
from spiral_engine import (
    PedigreeGenerator,
    PedigreeVisualizer,
    GenerationConfig,
    validate_logic,
    DNATableGenerator,
    Gene,
    InheritanceMode
)


app = Flask(__name__)
app.json.ensure_ascii = False  # 한글 출력


@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """가계도 생성 API"""
    try:
        data = request.get_json() or {}

        # 설정 파싱 (새로운 두 가족 구조)
        num_children_left = data.get('num_children_left', 2)
        num_children_right = data.get('num_children_right', 2)
        num_children_gen3 = data.get('num_children_gen3', 2)

        # 유전자 설정
        genes = [
            Gene(
                symbol="A",
                dominant_allele="A",
                recessive_allele="a",
                inheritance_mode=InheritanceMode.AUTOSOMAL_RECESSIVE,
                trait_name="(가)"
            ),
            Gene(
                symbol="B",
                dominant_allele="B",
                recessive_allele="b",
                inheritance_mode=InheritanceMode.AUTOSOMAL_RECESSIVE,
                trait_name="(나)"
            )
        ]

        config = GenerationConfig(
            num_children_left=num_children_left,
            num_children_right=num_children_right,
            num_children_gen3=num_children_gen3,
            ensure_both_genders=True,
            genes=genes
        )

        # 생성
        generator = PedigreeGenerator()
        family = generator.generate(config)

        # 검증
        is_valid, errors = validate_logic(family)

        if not is_valid:
            return jsonify({
                "success": False,
                "error": "논리 검증 실패",
                "details": errors
            }), 400

        # 시각화
        visualizer = PedigreeVisualizer()
        image_base64 = visualizer.get_base64_image(family)

        # DNA 표
        table_gen = DNATableGenerator()
        dna_table = table_gen.generate_table(family, hide_values=True)

        # 구성원 정보
        members_info = []
        for person in sorted(family.members.values(),
                           key=lambda p: (p.generation, p.x_pos)):
            traits = {}
            for gene in genes:
                trait = person.get_trait(gene.symbol)
                if trait:
                    traits[gene.trait_name] = {
                        "genotype": trait.genotype,
                        "affected": trait.is_affected()
                    }

            members_info.append({
                "id": person.id,
                "name": person.display_name,
                "gender": person.gender.value,
                "generation": person.generation,
                "traits": traits
            })

        return jsonify({
            "success": True,
            "image": image_base64,
            "members": members_info,
            "dna_table": dna_table,
            "validation": {
                "valid": True,
                "errors": []
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    print("=" * 50)
    print("Spiral Engine API Server")
    print("=" * 50)
    print("Server starting at http://localhost:5000")
    print()

    app.run(host='0.0.0.0', port=5000, debug=True)
