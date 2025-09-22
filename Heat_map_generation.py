# Ensure time is a datetime column
df_log['time'] = pd.to_datetime(df_log['time'], format='mixed')

# Get unique courses
unique_courses = df_log['courseid'].unique()

for course in unique_courses:
    # Remove slashes and backslashes from the course name
#    sanitized_course = re.sub(r'[\\/]', '', course)
    
    # Filter data for the current course
    course_data = df_log[df_log['courseid'] == course]
    
    # Step 1: Sort by user and time
    course_data = course_data.sort_values(by=['user', 'time'])

    # Step 2: Generate transitions
    course_data['next_section'] = course_data.groupby('user')['section (#1)'].shift(-1)
    transitions = course_data.dropna(subset=['next_section']).groupby(['section (#1)', 'next_section']).size().reset_index(name='count')

    # Step 3: Remove self-loops
    transitions = transitions[transitions['section (#1)'] != transitions['next_section']]

    # Step 4: Calculate total outgoing transitions for each source section
    outgoing_totals = transitions.groupby('section (#1)')['count'].sum().to_dict()

    # Step 5: Sort sections according to the sequence in the "section" column (Number (Integer))
    # Get unique section names ordered by their first appearance in the "section" column
    section_order_df = course_data[['section (#1)', 'section']].drop_duplicates().sort_values('section')
    section_order = section_order_df['section (#1)'].tolist()

    # If there are any sections in next_section not in section_order, append them at the end
    all_sections = list(section_order)
    for s in course_data['next_section'].dropna().unique():
        if s not in all_sections:
            all_sections.append(s)

    # Step 6: Create a transition matrix with the correct order
    transition_matrix = pd.DataFrame(0, index=all_sections, columns=all_sections)

    for _, row in transitions.iterrows():
        if outgoing_totals[row['section (#1)']] > 0:  # Normalize using total outgoing transitions
            transition_matrix.at[row['next_section'], row['section (#1)']] = (row['count'] / outgoing_totals[row['section (#1)']]) * 100

    # Step 7: Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=False, fmt=".1f", cmap="Blues", linewidths=0.5, linecolor='black')
    plt.title(f'Transition Frequency Heatmap for {course} (Normalized by Outgoing Transitions)', fontsize=16)
    plt.xlabel('From Section', fontsize=12)
    plt.ylabel('To Section', fontsize=12)
    plt.xticks(rotation=90, fontsize=10, ha='center')
    plt.yticks(rotation=0, fontsize=10, va='center')
    plt.tight_layout()
    plt.savefig(f"{course}.png")
    plt.close()
