---
title: "Tutorials"
layout: archive
collection: tutorials
permalink: /tutorials/
---


{% if paginator %}
  {% assign tutorials = paginator.tutorials %}
{% else %}
  {% assign tutorials = site.tutorials %}
{% endif %}

<script type="text/javascript">
    function filterTutorialsUsingCategory(selectedCategory) {
      {% for tutorial in tutorials %}
        var cats = {{ tutorial.tags | jsonify }}
        var tutorialDiv = document.getElementById("tutorial-{{tutorial.title | slugify}}");
        tutorialDiv.style.display = (selectedCategory == 'All' || cats.includes(selectedCategory))
          ? 'unset'
          : 'none';
      {% endfor %}
    }
</script>

  <div class="btn-group">
    <button id="All" class="button-71" role="button" onclick="filterTutorialsUsingCategory('All')">All ({{ tutorials.size }})</button>
    {% assign tags = site.tutorials | map: 'tags' | join: ','  | split: ','  | group_by: tag %}
    {% for cat in tags %} <!-- of the form {"name":"","items":[],"size":N}-->
      <button id="{{ cat.name }}" class="button-71" role="button" onclick="filterTutorialsUsingCategory(this.id)">{{ cat.name }} ({{ cat.size }})</button>
    {% endfor %}
    <hr />
  </div>
  <div class="tutorials-wrapper">
    {% for tutorial in tutorials %}
      {% assign post = tutorial %}
      <div class="tutorial" id="tutorial-{{tutorial.title | slugify}}">
        <p class="itemInteriorSection">
          {%- unless tutorial.hidden -%}
            {% include archive-single.html %}
            {% if tutorial.image %}
              <a href="{{ tutorial.link }}"><img src="{{ tutorial.image }}"></a>
            {% endif %}
          {%- endunless -%}
        </p>
      </div>
    {% endfor %}
  </div>
