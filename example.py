"""
Spiral Engine - 사용 예시
다양한 유전 문제 생성 시나리오
"""

from spiral_engine import (
    Gene, Family, Person, GeneticTrait,
    PedigreeGenerator, GenerationConfig,
    DNATableGenerator,
    PedigreeVisualizer, VisualizationConfig,
    LogicValidator,
    InheritanceMode
)
from spiral_engine.models import ChromosomeType, Gender, Phenotype
from spiral_engine.genetics import PedigreeConstraint


def example_1_basic_problem():
    """
    예시 1: 기본 상염색체 열성 유전 문제
    - 정상 부모에서 유전병 자녀가 태어나는 경우
    """
    print("\n" + "="*60)
    print("예시 1: 기본 상염색체 열성 유전")
    print("="*60)

    # 생성기 초기화
    generator = PedigreeGenerator(seed=42)

    # 유전자 정의
    gene_h = Gene(
        symbol='H',
        dominant_allele='H',
        recessive_allele='h',
        chromosome_type=ChromosomeType.AUTOSOMAL
    )

    # 가족 구조 생성
    family = generator.generate_family_structure()

    # 제약 조건: 2세대에서 최소 1명 발병
    constraints = [
        PedigreeConstraint(
            gene=gene_h,
            min_affected_count=1,
            affected_in_generation=2
        )
    ]

    # 역추적으로 유전자형 할당
    success, family = generator.assign_genotypes_with_backtracking(
        family, [gene_h], constraints
    )

    if success:
        print("\n✓ 문제 생성 성공!")

        # 가족 정보 출력
        print("\n【가족 구성원】")
        for person in family.all_members:
            trait = person.get_trait('H')
            if trait:
                print(f"  {person.display_name}: {trait.genotype} "
                      f"({trait.phenotype.value})"
                      f"{' [보인자]' if trait.is_carrier else ''}")

        # DNA 상대량 표 생성
        table_gen = DNATableGenerator()
        table = table_gen.generate_table(
            family, [gene_h],
            num_persons=4,
            difficulty='medium'
        )

        print("\n【DNA 상대량 표】")
        print(table.to_markdown())

        # 검증
        validator = LogicValidator()
        report = validator.validate_logic(family, [gene_h])
        print(f"\n【검증 결과】: {'✓ 통과' if report.is_valid else '✗ 실패'}")

    else:
        print("✗ 문제 생성 실패")


def example_2_xlinked_problem():
    """
    예시 2: X염색체 연관 열성 유전
    - 반성유전 패턴
    """
    print("\n" + "="*60)
    print("예시 2: X염색체 연관 열성 유전")
    print("="*60)

    generator = PedigreeGenerator(seed=123)

    # X-linked 유전자
    gene_t = Gene(
        symbol='T',
        dominant_allele='T',
        recessive_allele='t',
        chromosome_type=ChromosomeType.X_LINKED
    )

    # 문제 생성
    success, family = generator.generate_complete_problem(
        genes=[gene_t],
        mode=InheritanceMode.X_LINKED_RECESSIVE,
        min_affected=1
    )

    if success:
        print("\n✓ 문제 생성 성공!")

        print("\n【가족 구성원】")
        for person in family.all_members:
            trait = person.get_trait('T')
            if trait:
                gender_symbol = "♂" if person.gender == Gender.MALE else "♀"
                print(f"  {person.display_name} {gender_symbol}: {trait.genotype} "
                      f"({trait.phenotype.value})")

        # 검증
        validator = LogicValidator()
        report = validator.validate_logic(family, [gene_t])
        print(f"\n【검증 결과】: {'✓ 통과' if report.is_valid else '✗ 실패'}")

        # X-linked 특수 규칙 확인
        print("\n【X연관 유전 확인】")
        for person in family.all_members:
            if person.gender == Gender.MALE:
                trait = person.get_trait('T')
                if trait:
                    amounts = trait.get_dna_amounts()
                    total = sum(amounts.values())
                    print(f"  {person.display_name} (남성): DNA 총량 = {total}")


def example_3_multi_trait():
    """
    예시 3: 2형질 유전 문제
    - 상염색체 + X연관 복합
    """
    print("\n" + "="*60)
    print("예시 3: 2형질 유전 (상염색체 + X연관)")
    print("="*60)

    generator = PedigreeGenerator(seed=456)

    # 2형질 문제 생성
    success, family = generator.generate_multi_trait_problem(
        num_traits=2,
        modes=[
            InheritanceMode.AUTOSOMAL_RECESSIVE,
            InheritanceMode.X_LINKED_RECESSIVE
        ]
    )

    if success:
        print("\n✓ 문제 생성 성공!")

        print("\n【가족 구성원 - 2형질】")
        for person in family.all_members:
            traits_info = []
            for symbol, trait in person.traits.items():
                traits_info.append(f"{symbol}:{trait.genotype}")
            print(f"  {person.display_name}: {', '.join(traits_info)}")

        # DNA 상대량 표
        table_gen = DNATableGenerator()
        genes = family.genes
        table = table_gen.generate_table(
            family, genes,
            num_persons=4,
            difficulty='hard'
        )

        print("\n【DNA 상대량 표 (2형질)】")
        print(table.to_markdown())

        # 검증
        validator = LogicValidator()
        report = validator.validate_logic(family, genes)
        print(f"\n【검증 결과】: {'✓ 통과' if report.is_valid else '✗ 실패'}")


def example_4_visualization():
    """
    예시 4: 가계도 시각화
    """
    print("\n" + "="*60)
    print("예시 4: 가계도 시각화")
    print("="*60)

    generator = PedigreeGenerator(seed=789)

    # 문제 생성
    success, family = generator.generate_complete_problem(
        mode=InheritanceMode.AUTOSOMAL_RECESSIVE,
        min_affected=2
    )

    if success:
        print("\n✓ 문제 생성 성공!")

        # 시각화
        visualizer = PedigreeVisualizer(
            config=VisualizationConfig(
                fig_width=12,
                fig_height=8,
                show_genotype=True,
                show_carrier_pattern=True
            )
        )

        gene_symbol = family.genes[0].symbol if family.genes else None

        # 파일 저장
        visualizer.save_pedigree(
            family,
            filepath="output/example_pedigree.png",
            gene_symbol=gene_symbol,
            title="유전 가계도 예시"
        )
        print("✓ 가계도 이미지 저장: output/example_pedigree.png")


def example_5_custom_constraints():
    """
    예시 5: 사용자 정의 제약조건
    - 특정 패턴의 문제 생성
    """
    print("\n" + "="*60)
    print("예시 5: 사용자 정의 제약조건")
    print("="*60)

    generator = PedigreeGenerator(seed=101112)

    # 유전자 정의
    gene = Gene(
        symbol='R',
        dominant_allele='R',
        recessive_allele='r',
        chromosome_type=ChromosomeType.AUTOSOMAL
    )

    # 가족 구조 생성
    config = GenerationConfig(
        num_children_gen2=4,  # 자녀 4명
        ensure_both_genders_gen2=True
    )
    family = generator.generate_family_structure(config)

    # 제약조건: 자녀 중 2명 발병
    constraints = [
        PedigreeConstraint(
            gene=gene,
            min_affected_count=2,
            affected_in_generation=2
        )
    ]

    success, family = generator.assign_genotypes_with_backtracking(
        family, [gene], constraints
    )

    if success:
        print("\n✓ 문제 생성 성공!")

        # 자녀 정보 출력
        print("\n【자녀 세대】")
        children = family.get_generation(2)
        affected_count = 0
        for child in children:
            trait = child.get_trait('R')
            if trait:
                is_affected = trait.phenotype == Phenotype.RECESSIVE
                if is_affected:
                    affected_count += 1
                status = "발현" if is_affected else "정상"
                print(f"  {child.display_name}: {trait.genotype} ({status})")

        print(f"\n  → 발병자 수: {affected_count}명 (제약조건: 2명 이상)")

        # 검증
        validator = LogicValidator()
        report = validator.validate_logic(family, [gene])
        print(f"\n【검증 결과】: {'✓ 통과' if report.is_valid else '✗ 실패'}")


def main():
    """모든 예시 실행"""
    import os
    os.makedirs("output", exist_ok=True)

    print("\n" + "#"*60)
    print("# Spiral Engine - 사용 예시")
    print("#"*60)

    example_1_basic_problem()
    example_2_xlinked_problem()
    example_3_multi_trait()
    example_4_visualization()
    example_5_custom_constraints()

    print("\n" + "="*60)
    print("모든 예시 실행 완료!")
    print("="*60)


if __name__ == "__main__":
    main()
