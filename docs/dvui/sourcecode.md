.. meta::
:description: information on where to find the application source code

# Source Code

The Document Validation UI is built using [Vue.js version 2](https://v2.vuejs.org/).

The entry point of the application is the `main.js`, which loads all the Vue libraries and therefore opens the first Vue component, the `App.vue`. This component has the responsibility to tell the store to fetch the API data required to load the components. These files can be found under the `src` folder.

[This folder](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src) is the root of the project and is divided into:

### Assets

[The assets folder](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/assets) contains all the images and styles. Each component has its styles under the `SCSS` folder. The application uses the [Buefy](https://buefy.org) library, so if you want to edit the application theme you can go to the file `scss/variables.scss` and edit all the colors and variables to match the desired design.

The application uses FontAwesome for the icons. If you need to add new ones, you can do so on the `main.js` file on the icon library setup.

### Components

Vue components that are responsible for making the application work. They are divided into sections related to their function. The different components can be found [here](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/components).

### Directives

[Here](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/directives) are the directives for manipulating the elements in the HTML.

### Locales

In [this folder](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/locales) you can find the translation files, which consist of key-value pairs. There can be as many files as languages needed. Translations are implemented using the library [Vue I18n](https://vue-i18n.intlify.dev/), and the expected value is rendered by using the `$t` translation API and passing the key as an argument.

### Store

The [store](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/store) is implemented using [Vuex](https://vuex.vuejs.org/). It is responsible for saving all information coming from the API and local information regarding the use of the app, like the display scale. Most of the information is obtained, filtered, grouped, and processed in the store files which are separated by their concept.

### Utils

All utility functions are added [here](https://github.com/konfuzio-ai/document-validation-ui/tree/main/src/utils).

## **Tests**

You can test the application by running `npm run test:unit`. Tests are developed using [Vue Test Utils](https://github.com/vuejs/vue-test-utils) and are located in the `tests` folder. Mock data is used to get information into the testing environment, so there's no requirement to connect to an API. The tests are grouped by the different components and are focused on testing the components' behavior.
