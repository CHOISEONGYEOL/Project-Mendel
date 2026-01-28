"""
main.py - Spiral Engine 테스트 실행
수능 스타일 가계도 이미지 생성
"""

import os
from spiral_engine import (
    PedigreeGenerator,
    PedigreeVisualizer,
    GenerationConfig,
    GridConfig,
    validate_logic,
    DNATableGenerator,
    Gene,
    InheritanceMode
)


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("Spiral Engine - 수능 스타일 가계도 생성기")
    print("=" * 60)

    # 1. 유전자 설정
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

    # 2. 가계도 생성
    config = GenerationConfig(
        num_children_left=2,   # 왼쪽 조부모의 자녀 2명
        num_children_right=2,  # 오른쪽 조부모의 자녀 2명
        num_children_gen3=2,   # 3세대 자녀 2명
        ensure_both_genders=True,
        genes=genes
    )

    print("\n[1] 가계도 생성 중...")
    generator = PedigreeGenerator()
    family = generator.generate(config)

    # 3. 유전 논리 검증
    print("\n[2] 유전 논리 검증 중...")
    is_valid, errors = validate_logic(family)

    if is_valid:
        print("    [OK] 검증 통과!")
    else:
        print("    [FAIL] 검증 실패:")
        for error in errors:
            print(f"      - {error}")
        return

    # 4. 구성원 정보 출력
    print("\n[3] 구성원 정보:")
    print("-" * 50)

    for gen in [1, 2, 3]:
        members = family.get_generation(gen)
        print(f"\n  {gen}세대:")
        for person in sorted(members, key=lambda p: p.x_pos):
            gender_str = "남" if person.gender.value == "male" else "여"
            traits_str = []
            for gene in genes:
                trait = person.get_trait(gene.symbol)
                if trait:
                    status = "발현" if trait.is_affected() else "정상"
                    traits_str.append(f"{gene.trait_name}:{trait.genotype}({status})")

            print(f"    [{person.display_name}] {gender_str} | {', '.join(traits_str)}")

    # 5. 가계도 이미지 생성
    print("\n[4] 가계도 이미지 생성 중...")

    visualizer = PedigreeVisualizer()
    output_path = "pedigree_exam_style.png"
    visualizer.save_to_file(family, output_path)

    print(f"    [OK] 저장 완료: {output_path}")

    # 6. DNA 상대량 표 생성
    print("\n[5] DNA 상대량 표 생성 중...")

    table_gen = DNATableGenerator()
    table = table_gen.generate_table(family, hide_values=True)

    print("\n  표 헤더:", table["headers"])
    print("\n  표 데이터:")
    for row in table["rows"]:
        print(f"    {row}")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
