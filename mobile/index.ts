import { registerRootComponent } from 'expo';

import App from './App';
import { bootstrapMobileRuntime } from './src/runtime';

void bootstrapMobileRuntime();
registerRootComponent(App);
