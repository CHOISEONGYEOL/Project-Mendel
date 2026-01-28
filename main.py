"""
Spiral Engine - 수능 생명과학 유전 문제 생성기
메인 실행 파일

사용법:
    python main.py                    # 기본 문제 생성
    python main.py --mode x-linked    # X염색체 연관 유전 문제
    python main.py --traits 2         # 2형질 문제
    python main.py --difficulty hard  # 고난도 문제
"""

import argparse
import json
import os
from datetime import datetime
from typing import Optional

from spiral_engine import (
    Gene, Family,
    PedigreeGenerator, GenerationConfig,
    DNATableGenerator,
    PedigreeVisualizer, VisualizationConfig,
    LogicValidator,
    InheritanceMode
)
from spiral_engine.models import ChromosomeType, Phenotype


class SpiralEngine:
    """
    Spiral Engine 메인 클래스
    유전 문제 생성 및 관리
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 랜덤 시드 (재현성용)
        """
        self.generator = PedigreeGenerator(seed=seed)
        self.table_generator = DNATableGenerator()
        self.visualizer = PedigreeVisualizer()
        self.validator = LogicValidator()

    def generate_problem(
        self,
        mode: InheritanceMode = InheritanceMode.AUTOSOMAL_RECESSIVE,
        num_traits: int = 1,
        difficulty: str = 'medium',
        min_affected: int = 1
    ) -> dict:
        """
        새로운 유전 문제 생성

        Args:
            mode: 유전 방식
            num_traits: 형질 수 (1 또는 2)
            difficulty: 난이도 ('easy', 'medium', 'hard')
            min_affected: 최소 발병자 수

        Returns:
            문제 데이터 딕셔너리
        """
        print(f"\n{'='*50}")
        print("🧬 Spiral Engine - 유전 문제 생성 중...")
        print(f"{'='*50}")
        print(f"유전 방식: {mode.value}")
        print(f"형질 수: {num_traits}")
        print(f"난이도: {difficulty}")
        print()

        # 유전자 생성
        if num_traits == 1:
            success, family = self.generator.generate_complete_problem(
                mode=mode,
                min_affected=min_affected
            )
            genes = family.genes if success else []
        else:
            # 다형질 문제
            modes = [mode]
            if mode in [InheritanceMode.AUTOSOMAL_RECESSIVE,
                       InheritanceMode.AUTOSOMAL_DOMINANT]:
                modes.append(InheritanceMode.X_LINKED_RECESSIVE)
            else:
                modes.append(InheritanceMode.AUTOSOMAL_RECESSIVE)

            success, family = self.generator.generate_multi_trait_problem(
                num_traits=num_traits,
                modes=modes[:num_traits]
            )
            genes = family.genes if success else []

        if not success:
            print("❌ 문제 생성 실패. 다시 시도해주세요.")
            return {'success': False, 'error': '문제 생성 실패'}

        print("✓ 가계도 구조 생성 완료")
        print(f"  - 구성원 수: {len(family.all_members)}")

        # DNA 상대량 표 생성
        table = self.table_generator.generate_table(
            family, genes,
            num_persons=4,
            difficulty=difficulty
        )
        print("✓ DNA 상대량 표 생성 완료")

        # 논리 검증
        validation_report = self.validator.validate_logic(family, genes)
        print(f"✓ 논리 검증 완료: {'통과' if validation_report.is_valid else '실패'}")

        if not validation_report.is_valid:
            print("\n⚠️ 검증 오류:")
            for error in validation_report.get_errors():
                print(f"  - {error.message}")
            return {
                'success': False,
                'error': '논리 검증 실패',
                'validation': validation_report.to_dict()
            }

        # 문제 데이터 구성
        question_data = self.table_generator.create_question_data(
            table, family, genes
        )

        # 가계도 이미지 생성
        gene_symbol = genes[0].symbol if genes else None
        problem_img = self.visualizer.create_problem_image(
            family, gene_symbol,
            hide_genotypes=True,
            title="유전 가계도"
        )
        answer_img = self.visualizer.create_answer_image(
            family, gene_symbol,
            title="유전 가계도 (정답)"
        )

        print("✓ 가계도 이미지 생성 완료")

        # 결과 구성
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': mode.value,
                'num_traits': num_traits,
                'difficulty': difficulty
            },
            'family': self._family_to_dict(family),
            'genes': [self._gene_to_dict(g) for g in genes],
            'dna_table': question_data,
            'images': {
                'problem': problem_img,
                'answer': answer_img
            },
            'validation': validation_report.to_dict()
        }

        print(f"\n{'='*50}")
        print("✅ 문제 생성 완료!")
        print(f"{'='*50}")

        return result

    def _family_to_dict(self, family: Family) -> dict:
        """가족 객체를 딕셔너리로 변환"""
        members = []
        for person in family.all_members:
            member_data = {
                'id': person.id,
                'display_name': person.display_name,
                'gender': person.gender.value,
                'generation': person.generation,
                'father_id': person.father_id,
                'mother_id': person.mother_id,
                'spouse_id': person.spouse_id,
                'children_ids': person.children_ids,
                'traits': {}
            }

            for symbol, trait in person.traits.items():
                member_data['traits'][symbol] = {
                    'genotype': trait.genotype,
                    'phenotype': trait.phenotype.value,
                    'is_carrier': trait.is_carrier,
                    'dna_amounts': trait.get_dna_amounts()
                }

            members.append(member_data)

        return {
            'members': members,
            'couples': family.couples,
            'generations': {
                0: family.generation_0,
                1: family.generation_1,
                2: family.generation_2
            }
        }

    def _gene_to_dict(self, gene: Gene) -> dict:
        """유전자 객체를 딕셔너리로 변환"""
        return {
            'symbol': gene.symbol,
            'dominant_allele': gene.dominant_allele,
            'recessive_allele': gene.recessive_allele,
            'chromosome_type': gene.chromosome_type.value
        }

    def display_problem(self, result: dict):
        """문제를 콘솔에 표시"""
        if not result.get('success'):
            print(f"❌ 오류: {result.get('error')}")
            return

        print("\n" + "="*60)
        print("📋 생성된 문제")
        print("="*60)

        # 가족 구성원 정보
        print("\n【가족 구성원】")
        family_data = result['family']
        for member in family_data['members']:
            traits_str = ", ".join(
                f"{s}: {t['genotype']} ({t['phenotype']})"
                for s, t in member['traits'].items()
            )
            print(f"  {member['display_name']} ({member['id']}): "
                  f"{member['gender']} - {traits_str}")

        # DNA 상대량 표
        print("\n【DNA 상대량 표】")
        print(result['dna_table']['table_markdown'])

        # 정답
        print("\n【정답】")
        print(result['dna_table']['answer_markdown'])

        # 힌트
        if result['dna_table'].get('question_hints'):
            print("\n【힌트】")
            for hint in result['dna_table']['question_hints']:
                print(f"  • {hint}")

    def save_problem(self, result: dict, output_dir: str = "output"):
        """문제를 파일로 저장"""
        if not result.get('success'):
            print("❌ 저장할 문제가 없습니다.")
            return

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"problem_{timestamp}"

        # JSON 데이터 저장 (이미지 제외)
        json_data = {k: v for k, v in result.items() if k != 'images'}
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON 저장: {json_path}")

        # 이미지 저장
        import base64

        if result.get('images'):
            # 문제 이미지
            problem_img_path = os.path.join(output_dir, f"{base_name}_problem.png")
            with open(problem_img_path, 'wb') as f:
                f.write(base64.b64decode(result['images']['problem']))
            print(f"✓ 문제 이미지 저장: {problem_img_path}")

            # 정답 이미지
            answer_img_path = os.path.join(output_dir, f"{base_name}_answer.png")
            with open(answer_img_path, 'wb') as f:
                f.write(base64.b64decode(result['images']['answer']))
            print(f"✓ 정답 이미지 저장: {answer_img_path}")


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Spiral Engine - 수능 생명과학 유전 문제 생성기"
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='autosomal_recessive',
        choices=['autosomal_recessive', 'autosomal_dominant',
                 'x_linked_recessive', 'x_linked_dominant'],
        help="유전 방식 (기본: autosomal_recessive)"
    )

    parser.add_argument(
        '--traits', '-t',
        type=int,
        default=1,
        choices=[1, 2],
        help="형질 수 (1 또는 2, 기본: 1)"
    )

    parser.add_argument(
        '--difficulty', '-d',
        type=str,
        default='medium',
        choices=['easy', 'medium', 'hard'],
        help="난이도 (기본: medium)"
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help="랜덤 시드 (재현성용)"
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help="출력 디렉토리 (기본: output)"
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help="문제를 파일로 저장"
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help="콘솔 출력 생략"
    )

    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 유전 방식 매핑
    mode_map = {
        'autosomal_recessive': InheritanceMode.AUTOSOMAL_RECESSIVE,
        'autosomal_dominant': InheritanceMode.AUTOSOMAL_DOMINANT,
        'x_linked_recessive': InheritanceMode.X_LINKED_RECESSIVE,
        'x_linked_dominant': InheritanceMode.X_LINKED_DOMINANT,
    }

    mode = mode_map.get(args.mode, InheritanceMode.AUTOSOMAL_RECESSIVE)

    # 엔진 초기화
    engine = SpiralEngine(seed=args.seed)

    # 문제 생성
    result = engine.generate_problem(
        mode=mode,
        num_traits=args.traits,
        difficulty=args.difficulty
    )

    # 출력
    if not args.no_display:
        engine.display_problem(result)

    # 저장
    if args.save:
        engine.save_problem(result, args.output)


if __name__ == "__main__":
    main()
