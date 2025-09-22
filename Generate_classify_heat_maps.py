import pandas as pd

log=pd.read_csv(r"csv_log.py")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import defaultdict

log['time'] = pd.to_datetime(log['time'], format='mixed')

# Initialize pattern counter
pattern_counts = defaultdict(int)
pattern_details = defaultdict(list)


# Define pattern detection functions
def calculate_diagonal_strength(matrix, offset=1):
    """Calculate the strength of a diagonal in the transition matrix"""
    diagonal = np.diagonal(matrix.values, offset=offset)
    if len(diagonal) == 0:
        return 0

    # Count cells with meaningful transitions (>30%)
    significant_count = np.sum(diagonal > 30)

    # Density = proportion of diagonal cells that are significant
    return significant_count / len(diagonal)


def detect_dominant_section(matrix, relative_threshold=3.0, min_sources=2):
    """Detect if there's a dominant section that receives many transitions"""
    # Row sums = incoming transitions to each section
    row_sums = matrix.sum(axis=1)

    if len(row_sums) == 0:
        return False, None

    max_row = row_sums.max()
    mean_row = row_sums.mean()

    if mean_row == 0:
        return False, None

    # Count how many sections send significant traffic (>5%) to this section
    dominant_idx = row_sums.idxmax()
    if isinstance(dominant_idx, (pd.Series, np.ndarray, list)):
        dominant_idx = dominant_idx[0]

    significant_sources = (matrix.loc[dominant_idx] > 5).sum() if isinstance(matrix.loc[dominant_idx], pd.Series) else (matrix.loc[dominant_idx].iloc[0] > 5).sum()
    # A section is dominant if it receives significantly more than average
    # and from multiple sources
    is_dominant = (max_row > mean_row * relative_threshold) and (significant_sources >= min_sources)

    return is_dominant, dominant_idx


def calculate_entropy(matrix):
    """Calculate entropy of the transition matrix"""
    # Convert to probabilities
    prob_matrix = matrix / 100
    flat = prob_matrix.values.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return 0
    flat = flat / flat.sum()  # Normalize
    return -np.sum(flat * np.log2(flat + 1e-10))


def classify_navigation_pattern_alphabetical(transition_matrix):
    n = len(transition_matrix)
    section_names = transition_matrix.index.tolist()

    main_diagonal = calculate_diagonal_strength(transition_matrix, offset=-1)
    prev_diagonal = calculate_diagonal_strength(transition_matrix, offset=1)
    blended_score = calculate_diagonal_strength(transition_matrix, offset=-2)
    has_dominant, dominant_section = detect_dominant_section(transition_matrix)
    entropy = calculate_entropy(transition_matrix)
    total_main_diagonals = main_diagonal + prev_diagonal

    if blended_score > 0.3:
        pattern = 'blended mode'
    elif has_dominant and total_main_diagonals < 0.6:
        pattern = 'dominant_section'
    elif has_dominant and (total_main_diagonals >= 0.6 or blended_score > 0.1):
        pattern = 'combination'
    elif main_diagonal > 0.3 and prev_diagonal >= 0.3 :
        pattern = 'double_diagonal'
    elif main_diagonal > 0.4 and prev_diagonal < 0.3:
        pattern = 'single_diagonal'
    elif prev_diagonal > 0.4 and main_diagonal < 0.15:
        pattern = 'single_diagonal'
    elif entropy > 6.5:
        pattern = 'no clear pattern'
    else:
        pattern = 'no clear pattern'

    return {
        'pattern': pattern,
        'main_diagonal': main_diagonal,
        'prev_diagonal': prev_diagonal,
        'blended_score': blended_score,
        'total_diagonals': total_main_diagonals,
        'entropy': entropy,
        'has_dominant': has_dominant,
        'dominant_section': dominant_section if has_dominant else None
    }


# Get unique courses
unique_courses = log['courseid'].unique()

# Store results for each course
course_patterns = []

# Iterate over each course
for courseid in unique_courses:
#    sanitized_course = re.sub(r'[\\/]', '', course)
    course_data = log[log['courseid'] == courseid]

    # Step 1: Sort by user and time
    course_data = course_data.sort_values(by=['user', 'time'])

    # Step 2: Generate transitions
    course_data['next_section'] = course_data.groupby('user')['section (#1)'].shift(-1)
    transitions = course_data.dropna(subset=['next_section']).groupby(
        ['section (#1)', 'next_section']).size().reset_index(name='count')

    # Step 3: Remove self-loops
    transitions = transitions[transitions['section (#1)'] != transitions['next_section']]

    # Skip if no transitions
    if len(transitions) == 0:
        continue

    # Step 4: Calculate total outgoing transitions for each source section
    outgoing_totals = transitions.groupby('section (#1)')['count'].sum().to_dict()

        # Step 5: Create a transition matrix
        # IMPORTANT: Use course order, not alphabetical
    section_order_df = course_data[['section (#1)', 'section']].drop_duplicates().sort_values('section')
    all_sections = section_order_df['section (#1)'].tolist()
    all_sections = list(all_sections)

    transition_matrix = pd.DataFrame(0, index=all_sections, columns=all_sections)
    print(transition_matrix)
    for _, row in transitions.iterrows():
        if outgoing_totals[row['section (#1)']] > 0:
            transition_matrix.at[row['next_section'], row['section (#1)']] = (row['count'] / outgoing_totals[
                row['section (#1)']]) * 100




    # Classify the pattern
    pattern_info = classify_navigation_pattern_alphabetical(transition_matrix)
    pattern_info['course'] = courseid
    pattern_info['num_sections'] = len(all_sections)
    pattern_info['num_students'] = course_data['user'].nunique()
    pattern_info['num_transitions'] = transitions['count'].sum()

    # Store results
    course_patterns.append(pattern_info)
    pattern_counts[pattern_info['pattern']] += 1
    pattern_details[pattern_info['pattern']].append(courseid)

    # Step 6: Generate heatmap with pattern annotation
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=False, fmt=".1f", cmap="Blues", linewidths=0.5, linecolor='black')
    plt.title(f'Transition Frequency Heatmap for {courseid}\nPattern: {pattern_info["pattern"]}', fontsize=16)
    plt.xlabel('From Section', fontsize=12)
    plt.ylabel('To Section', fontsize=12)
    plt.xticks(rotation=90, fontsize=10, ha='center')
    plt.yticks(rotation=0, fontsize=10, va='center')
    plt.tight_layout()

    # Add pattern metrics to the plot
    metrics_text = f"Main diag: {pattern_info['main_diagonal']:.2f}, Prev diag: {pattern_info['prev_diagonal']:.2f}\n"
    metrics_text += f"Entropy: {pattern_info['entropy']:.2f}"
    if pattern_info['has_dominant']:
        metrics_text += f"\nDominant: {pattern_info['dominant_section']}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.savefig(f"{courseid}_{pattern_info['pattern']}.png")
    plt.close()

# Create summary report
results_df = pd.DataFrame(course_patterns)

# Generate summary statistics
print("\n=== NAVIGATION PATTERN DISTRIBUTION ===")
print(f"Total courses analyzed: {len(course_patterns)}")
print("\nPattern counts:")
for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(course_patterns)) * 100
    print(f"{pattern}: {count} ({percentage:.1f}%)")

# Detailed statistics by pattern
if len(results_df) > 0:
    summary_stats = results_df.groupby('pattern').agg({
        'course': 'count',
        'entropy': ['mean', 'std'],
        'main_diagonal': ['mean', 'std'],
        'prev_diagonal': ['mean', 'std'],
        'total_diagonals': 'mean',
        'num_sections': 'mean',
        'num_students': 'mean',
        'num_transitions': 'mean'
    }).round(3)

    print("\n=== PATTERN CHARACTERISTICS ===")
    print(summary_stats)

    # Export detailed results
    results_df.to_csv('course_pattern_classification.csv', index=False)

    # Create a summary visualization
    plt.figure(figsize=(12, 6))
    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    bars = plt.bar(patterns, counts, color=colors[:len(patterns)])

    plt.title('Distribution of Navigation Patterns Across Courses', fontsize=14)
    plt.xlabel('Pattern Type', fontsize=12)
    plt.ylabel('Number of Courses', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add percentage labels on bars
    for bar, count in zip(bars, counts):
        percentage = (count / len(course_patterns)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('pattern_distribution.png', dpi=300)
    plt.close()

    # Create detailed pattern examples report
    print("\n=== EXAMPLE COURSES PER PATTERN ===")
    for pattern, courses in pattern_details.items():
        print(f"\n{pattern} ({len(courses)} courses):")
        for course in courses[:5]:
            print(f"  - {course}")

    # Print debug information for a few courses
    print("\n=== DEBUG: First 5 courses metrics ===")
    for i, row in results_df.head().iterrows():
        print(f"\nCourse: {row['course'][:40]}...")
        print(f"  Pattern: {row['pattern']}")
        print(f"  Main diagonal: {row['main_diagonal']:.3f}")
        print(f"  Prev diagonal: {row['prev_diagonal']:.3f}")
        print(f"  Total diagonals: {row['total_diagonals']:.3f}")
        print(f"  Entropy: {row['entropy']:.3f}")
