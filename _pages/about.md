---
layout: about
title: about
permalink: /
subtitle: <a href='#'>Machine Learning Engineer</a>

profile:
  align: right
  image: about.jpg
  image_circular: false # crops the image to make it circular
  address: >

news: false  # includes a list of news items
social: true  # includes social icons at the bottom of the page
---

<H3>Expericia Docet</H3>

Hi,
I'm a AI & Data Center GPU Solutions Architect at AMD.
I'm passionated in AI, Accelerated Heterogeneous Computing, HPC, and GPU programming.

Previously, I was a LLM engineer at NAVER and Solutions Architect at NVIDIA.
I also authored a CUDA programming book - Learn CUDA programming - with Bharatkumar Sharma published from Packt pub.

<!-- CV -->
<details>
  <article>
    <div class="cv">
      {% for entry in site.data.cv %}
        <div class="card mt-3 p-3">
          <h3 class="card-title font-weight-medium">{{ entry.title }}</h3>
          <div>
          {% if entry.type == "list" %}
            {% include cv/list.html %}
          {% elsif entry.type == "map" %}
            {% include cv/map.html %}
          {% elsif entry.type == "nested_list" %}
            {% include cv/nested_list.html %}
          {% elsif entry.type == "time_table" %}
          {% include cv/time_table.html %}
          {% else %}
            {{ entry.contents }}
          {% endif %}
          </div>
        </div>
      {% endfor %}
      </div>
  </article>
</details>
