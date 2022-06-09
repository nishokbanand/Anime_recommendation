<template>
  <div>
    <Animes :animes="myjson" />
  </div>
</template>

<script>
import Animes from "./components/Animes.vue";
export default {
  data() {
    return {
      myjson: [],
    };
  },
  async created() {
    this.myjson = await this.fetchAnimes();
  },
  methods: {
    async fetchAnimes() {
      const res = await fetch("http://localhost:5000/animes");
      const data = await res.json();
      return data;
    },
  },
  components: {
    Animes,
  },
};
</script>
<style>
</style>
