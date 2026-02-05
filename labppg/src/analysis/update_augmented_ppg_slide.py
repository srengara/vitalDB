#!/usr/bin/env python3
"""Update Augmented PPG Track Timeline slide with new dates and training schedule."""

with open('INVESTOR_PRESENTATION_FINALv1.0.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Augmented PPG Track slide
slide_start = content.find('<!-- SLIDE 13: Augmented PPG Track Timeline -->')
slide_end = content.find('<!-- SLIDE 14: Current Status -->')

if slide_start == -1 or slide_end == -1:
    print("ERROR: Could not find slide markers")
    exit(1)

# Create the new slide content
new_slide = '''<!-- SLIDE 13: Augmented PPG Track Timeline -->
<!-- =============================================== -->
<div class="slide">
    <h1 class="section-header">Augmented PPG Track: FSR Multi-Modal Training</h1>

    <div class="timeline">
        <div class="timeline-item">
            <div class="timeline-date"><strong>Feb 2</strong><br><span class="badge badge-info">WEEK 1</span></div>
            <div class="timeline-content">
                <h4>Initial Dataset Available</h4>
                <p><strong>50 FSR cases</strong> delivered | Begin model training with first batch</p>
            </div>
        </div>

        <div class="timeline-item">
            <div class="timeline-date"><strong>Feb 9</strong><br><span class="badge badge-info">WEEK 2</span></div>
            <div class="timeline-content">
                <h4>Batch 2: Re-train</h4>
                <p><strong>+50 cases (Total: 100)</strong> | Re-train model with expanded dataset</p>
            </div>
        </div>

        <div class="timeline-item">
            <div class="timeline-date"><strong>Feb 16</strong><br><span class="badge badge-info">WEEK 3</span></div>
            <div class="timeline-content">
                <h4>Batch 3: Re-train</h4>
                <p><strong>+50 cases (Total: 150)</strong> | Continue incremental training</p>
            </div>
        </div>

        <div class="timeline-item">
            <div class="timeline-date"><strong>Feb 23</strong><br><span class="badge badge-success">WEEK 4</span></div>
            <div class="timeline-content">
                <h4>Batch 4: Final Training & TRL 4 Validation</h4>
                <p><strong>+50 cases (Total: 200 FSR cases)</strong> | Re-train with complete dataset</p>
            </div>
        </div>

        <div class="timeline-item">
            <div class="timeline-date"><strong>Last Week of Feb</strong><br><span class="badge badge-warning">MILESTONE</span></div>
            <div class="timeline-content">
                <h4>TRL 4 Achievement</h4>
                <p><strong>Target: 70% accuracy</strong> on 200 FSR cases | Technology Readiness Level 4 validated</p>
            </div>
        </div>
    </div>

    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.5rem; border-radius: 1rem; margin-top: 2rem; border-left: 6px solid #10b981;">
        <h4 style="color: #065f46; margin-bottom: 0.75rem;">Training Strategy</h4>
        <div class="grid-2" style="gap: 2rem;">
            <div>
                <p style="color: #065f46; margin: 0; font-size: 1.05rem;">
                    <strong>Incremental Training:</strong> Model re-trained weekly with cumulative data (50 → 100 → 150 → 200 cases) to leverage transfer learning and maximize performance
                </p>
            </div>
            <div>
                <p style="color: #065f46; margin: 0; font-size: 1.05rem;">
                    <strong>TRL 4 Target:</strong> 70% prediction accuracy on FSR track validates technology feasibility in lab environment (Technology Readiness Level 4)
                </p>
            </div>
        </div>
    </div>

    <div class="grid-2" style="margin-top: 1.5rem; gap: 1.5rem;">
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 1rem; border-left: 6px solid #f59e0b;">
            <p style="color: #92400e; margin: 0; font-size: 1.05rem;"><strong>FSR Track Focus:</strong> Force Sensitive Resistor PPG - primary signal for February validation</p>
        </div>
        <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 1.5rem; border-radius: 1rem; border-left: 6px solid #3b82f6;">
            <p style="color: #1e40af; margin: 0; font-size: 1.05rem;"><strong>Weekly Cadence:</strong> 50 new cases every week ensures continuous model improvement</p>
        </div>
    </div>

    <div class="slide-number">12</div>
</div>

<!-- =============================================== -->
<!-- '''

# Replace the slide
content = content[:slide_start] + new_slide + content[slide_end:]

# Write back
with open('INVESTOR_PRESENTATION_FINALv1.0.html', 'w', encoding='utf-8') as f:
    f.write(content)

print('Successfully updated Augmented PPG Track Timeline slide')
print('New timeline:')
print('  - Feb 2: 50 FSR cases available, start training')
print('  - Feb 9: +50 cases (100 total), re-train')
print('  - Feb 16: +50 cases (150 total), re-train')
print('  - Feb 23: +50 cases (200 total), re-train')
print('  - Last week of Feb: TRL 4 with 70% accuracy target')
