import knime.scripting.io as knio

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

df_log = knio.input_tables[0].to_pandas()

# Ensure time is a datetime column
df_log['time'] = pd.to_datetime(df_log['time'], format='mixed')


    # Step 1: Sort by user and time
course_data = df_log.sort_values(by=['student_course_id', 'time'])

    # Step 2: Generate transitions
course_data['next_section'] = course_data.groupby('student_course_id')['section'].shift(-1)
course_data['section'] = course_data['section'].astype('Int64')
course_data['next_section'] = course_data['next_section'].astype('Int64')
transitions = course_data.dropna(subset=['next_section']).groupby(['section', 'next_section']).size().reset_index(name='count')

    # Step 3: Remove self-loops
transitions = transitions[transitions['section'] != transitions['next_section']]

    # Step 4: Calculate total outgoing transitions for each source section
outgoing_totals = transitions.groupby('section')['count'].sum().to_dict()

    # Step 5: Sort sections according to the sequence in the "section" column (Number (Integer))
    # Get unique section names ordered by their first appearance in the "section" column
section_order_df = course_data[['section']].drop_duplicates().sort_values('section')
section_order = section_order_df['section'].tolist()

    # If there are any sections in next_section not in section_order, append them at the end
all_sections = list(section_order)
for s in course_data['next_section'].dropna().unique():
    if s not in all_sections:
        all_sections.append(s)

    # Step 6: Create a transition matrix with the correct order
transition_matrix = pd.DataFrame(0, index=all_sections, columns=all_sections)
for _, row in transitions.iterrows():
    if outgoing_totals[row['section']] > 0:  # Normalize using total outgoing transitions
        transition_matrix.at[row['next_section'], row['section']] = (row['count'] / outgoing_totals[row['section']]) * 100


    # Step 7: Generate heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=False, fmt=".1f", cmap="Blues", linewidths=0.5, linecolor='black')
plt.xlabel('From Section', fontsize=12)
plt.ylabel('To Section', fontsize=12)
plt.xticks(rotation=90, fontsize=10, ha='center')
plt.yticks(rotation=0, fontsize=10, va='center')
plt.tight_layout()
plt.savefig(f"cross-course.png")
plt.close()

knio.output_tables[0] = knio.input_tables[0]
